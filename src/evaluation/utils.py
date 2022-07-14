from moses.utils import disable_rdkit_log, enable_rdkit_log
from moses.utils import mapper, get_mol
from moses.metrics.metrics import fraction_valid, fraction_unique, \
    remove_invalid, novelty, internal_diversity, fraction_passes_filters, cos_similarity
from moses.metrics.utils import logP, SA, NP, QED, weight, fingerprints, \
    average_agg_tanimoto, compute_scaffolds, compute_fragments
from multiprocessing import Pool
import numpy as np
import logging
import random
from fcd_torch import FCD as FCDMetric

logger = logging.getLogger(__name__)


def calc_fcd(generated_smiles, ref_smiles,  batch_size=512, n_jobs=1, device='cpu'):
    ref_stats = {}
    kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
    ref_stats['FCD'] = FCDMetric(**kwargs_fcd).precalc(ref_smiles)
    kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
    return FCDMetric(**kwargs_fcd)(gen=generated_smiles, pref=ref_stats['FCD'])


def calc_snn(generated_mols, ref_mols, fp_type='morgan', n_jobs=1, device='cpu'):
    gen_fps = fingerprints(generated_mols, n_jobs=n_jobs, fp_type=fp_type)
    ref_fps = fingerprints(ref_mols, n_jobs=n_jobs, fp_type=fp_type)
    return average_agg_tanimoto(ref_fps, gen_fps, device=device)


def calc_scaffolds(generated_mols, ref_mols, n_jobs=1):
    gen_scaf = compute_scaffolds(generated_mols, n_jobs=n_jobs)
    ref_scaf = compute_scaffolds(ref_mols, n_jobs=n_jobs)
    return cos_similarity(gen_scaf, ref_scaf)


def calc_fragments(generated_mols, ref_mols, n_jobs=1):
    gen_frag = compute_fragments(generated_mols, n_jobs=n_jobs)
    ref_frag = compute_fragments(ref_mols, n_jobs=n_jobs)
    return cos_similarity(gen_frag, ref_frag)


def calc_test_metrics(generated, reference, train, n_jobs=1, device='cpu'):
    close_pool = False
    disable_rdkit_log()
    if n_jobs != 1:
        pool = Pool(n_jobs)
        close_pool = True
    else:
        pool = 1
    generated = remove_invalid(generated, canonize=True)
    if len(generated) == 0:
        logging.warning('Number of valid molecules 0')
        return {}
    gen_mols = mapper(pool)(get_mol, generated)
    ref_mols = mapper(pool)(get_mol, reference)
    metrics = {}
    try:
        metrics['IntDiv'] = internal_diversity(gen_mols, n_jobs=pool, device=device)
        metrics['IntDiv2'] = internal_diversity(gen_mols, n_jobs=pool, device=device, p=2)
        metrics['Filters'] = fraction_passes_filters(gen_mols, n_jobs=pool)
        metrics['Scaf'] = calc_scaffolds(gen_mols, ref_mols, n_jobs=pool)
        metrics['Frag'] = calc_fragments(gen_mols, ref_mols, n_jobs=pool)
        metrics['SNN'] = calc_snn(gen_mols, ref_mols, n_jobs=pool, device=device)
        metrics['FCD'] = calc_fcd(generated, reference, n_jobs=n_jobs, device=device)
        metrics['novelty'] = novelty(gen_mols, train, n_jobs=pool)
    except Exception as e:
        logger.warning(e)
    if close_pool:
        pool.close()
        pool.join()
    return metrics


def calc_val_metrics(gen, device='cuda',  n_jobs=1):
    close_pool = False
    disable_rdkit_log()
    if n_jobs != 1:
        pool = Pool(n_jobs)
        close_pool = True
    else:
        pool = 1
    logger.info(f'Calculating metrics for {len(gen)} molecules')
    metrics = {}
    metrics['valid'] = fraction_valid(gen, n_jobs=pool)
    gen = remove_invalid(gen, canonize=True)
    logger.info(f'Number of valid molecules: {len(gen)}')
    if len(gen) == 0:
        return metrics
    metrics['unique'] = fraction_unique(gen, len(gen), pool)
    logger.info(f'Uniqueness {metrics["unique"]}')
    logger.info(f'Calculating properties..')
    try:
        mols = mapper(pool)(get_mol, gen)
        logP_scores = mapper(pool)(logP, mols)
        metrics['logP'] = np.mean(logP_scores)
        metrics['logP_std'] = np.std(logP_scores)

        SA_scores = mapper(pool)(SA, mols)
        metrics['SA'] = np.mean(SA_scores)
        metrics['SA_std'] = np.std(SA_scores)

        NP_scores = mapper(pool)(NP, mols)
        metrics['NP'] = np.mean(NP_scores)
        metrics['NP_std'] = np.std(NP_scores)

        QED_scores = mapper(pool)(QED, mols)
        metrics['QED'] = np.mean(QED_scores)
        metrics['QED_std'] = np.std(QED_scores)

        weight_scores = mapper(pool)(weight, mols)
        metrics['weight'] = np.mean(weight_scores)
        metrics['weight_std'] = np.std(weight_scores)
        #for name, func in [('logP', logP), ('SA', SA),
        #                   ('QED', QED), ('NP', NP)
        #                   ('weight', weight)]:
        #    scores = mapper(pool)(func, mols)
        #    metrics[name] = np.mean(scores)
        #    metrics[name + '_std'] = np.std(scores)

        metrics['Filters'] = fraction_passes_filters(mols, pool)

        div_mols = random.sample(mols, min(1000, len(gen)))
        metrics['IntDiv'] = internal_diversity(div_mols, pool, device=device)
        metrics['IntDiv2'] = internal_diversity(div_mols, pool, device=device, p=2)
        enable_rdkit_log()
        if close_pool:
            pool.close()
            pool.join()
    except Exception as err:
        logger.info(err)
    return metrics




