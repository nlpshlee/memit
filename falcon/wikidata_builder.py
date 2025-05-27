from .data_preprocessor import *
from .cf_pair import *
from .wiki_util import *

from tqdm import tqdm


SEED = 7
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def _print_and_write(out_path: str, file_name: str, prefix: str, data_set: set=None, data_dict: dict=None):
    if data_set is not None:
        print(f'# wikidata_builder._print_and_write() [{file_name}] {prefix}_set size : {len(data_set)}')
        out_file_path = f'{out_path}/{file_name}/{file_name}_{prefix}_set.txt'
        write_set(data_set, out_file_path)
    if data_dict is not None:
        print(f'# wikidata_builder._print_and_write() [{file_name}] {prefix}_dict size : {len(data_dict)}')
        out_file_path = f'{out_path}/{file_name}/{file_name}_{prefix}_dict.json'
        serializable = { key: sorted(list(value)) for key, value in sorted_dict_key(data_dict).items() }
        write_json_to_file(serializable, out_file_path)


def make_cf_infos(in_file_paths: List[str], out_path: str):
    subject_set_all = set()
    relation_dict_all = dict()
    object_set_all = set()
    object_org_set_all = set()
    object_new_set_all = set()
    triple_set_all = set()

    for in_file_path in in_file_paths:
        file_name = get_file_name(in_file_path)

        datas = load_datas(in_file_path)
        subject_set = set()
        relation_dict = dict()
        object_set = set()
        object_org_set = set()
        object_new_set = set()
        triple_set = set()

        for data in datas:
            cf_pair = CFPair(data)

            if cf_pair.validate():            
                subject = cf_pair._subject
                relation_id = cf_pair._relation_id
                relation = cf_pair._prompt
                object_org = cf_pair._target_true_str
                object_new = cf_pair._target_new_str

                subject_set.add(subject)
                add_dict_set(relation_dict, relation_id, [relation])
                object_set.add(object_org)
                object_set.add(object_new)
                object_org_set.add(object_org)
                object_new_set.add(object_new)
                triple_set.add(f'{subject}\t{relation_id}\t{relation}\t{object_org}')

                subject_set_all.add(subject)
                add_dict_set(relation_dict_all, relation_id, [relation])
                object_set_all.add(object_org)
                object_set_all.add(object_new)
                object_org_set_all.add(object_org)
                object_new_set_all.add(object_new)
                triple_set_all.add(f'{subject}\t{relation_id}\t{relation}\t{object_org}')
        
        _print_and_write(out_path, file_name, 'subject', data_set=subject_set)
        _print_and_write(out_path, file_name, 'relation', data_dict=relation_dict)
        _print_and_write(out_path, file_name, 'object', data_set=object_set)
        _print_and_write(out_path, file_name, 'object_org', data_set=object_org_set)
        _print_and_write(out_path, file_name, 'object_new', data_set=object_new_set)
        _print_and_write(out_path, file_name, 'triple', data_set=triple_set)
    
    _print_and_write(out_path, 'all', 'subject', data_set=subject_set_all)
    _print_and_write(out_path, 'all', 'relation', data_dict=relation_dict_all)
    _print_and_write(out_path, 'all', 'object', data_set=object_set_all)
    _print_and_write(out_path, 'all', 'object_org', data_set=object_org_set_all)
    _print_and_write(out_path, 'all', 'object_new', data_set=object_new_set_all)
    _print_and_write(out_path, 'all', 'triple', data_set=triple_set_all)


def make_triple_with_wiki(subject_labels: list, relation_pids: list, limit: int, sparql=None, do_print=False):
    results = []
    subject_idx = -1
    relation_cnt_freq = {}

    for subject_label in tqdm(subject_labels):
        relations = []

        for relation_pid in relation_pids:
            if do_print:
                print(f'\nsubject_label : {subject_label}, relation_pid : {relation_pid}')

            # Wiki 검색
            object_labels, object_pids = get_object_of_subject_and_relation(subject_label, relation_pid, limit, sparql=sparql)

            if object_labels is not None and len(object_labels) > 0:
                objects = []

                for i, object_label in enumerate(object_labels):
                    object_pid = object_pids[i].split('/')[-1]

                    if do_print:
                        print(f'[{i}] object_label : {object_label}, object_pid : {object_pid}')
                    
                    objects.append({
                        'object_label': object_label,
                        'object_pid': object_pid
                    })
                
                relations.append({
                    'relation_pid': relation_pid,
                    'object_cnt': len(objects),
                    'objects': objects
                })
            
            if len(relations) == 10:
                break
        
        relation_cnt = len(relations)
        if relation_cnt > 1:
            subject_idx += 1

            results.append({
                'subject_idx': subject_idx,
                'subject': subject_label,
                'relation_cnt': relation_cnt,
                'relations': relations
            })
        
        if not relation_cnt in relation_cnt_freq.keys():
            relation_cnt_freq[relation_cnt] = 1
        else:
            relation_cnt_freq[relation_cnt] += 1

        # if len(results) == 10:
        #     break
        
        if len(results) == 10:
            print(f'# wikidata_builder.make_triple_with_wiki() making count : {len(results)}')
    print(f'# wikidata_builder.make_triple_with_wiki() making count : {len(results)}\n')

    return results, relation_cnt_freq


def run_250523_make_triple_with_wiki(out_path: str, limit, start=-1, end=-1):
    sparql = get_sparql()

    subject_set = set()
    in_file_path = f'{out_path}/all/all_subject_set.txt'
    file_to_set(in_file_path, subject_set)
    print(f'\nsubject_set size : {len(subject_set)}')

    if start >= len(subject_set):
        print(f'# (error) [{start}] is out of range #')
        return None

    in_file_path = f'{out_path}/all/all_relation_dict.json'
    relation_dict = load_json_file(in_file_path)
    print(f'relation_dict size : {len(relation_dict)}\n')

    subject_labels = sorted(list(subject_set))
    relation_pids = sorted(list(relation_dict.keys()))

    if start == -1 or end == -1:
        out_file_path1 = f'{out_path}/triple_with_wiki.json'
        out_file_path2 = f'{out_path}/triple_with_wiki_count.txt'
    else:
        print(f'start : {start}, end : {end-1}\n')
        subject_labels = subject_labels[start:min(end, len(subject_labels))]
        out_file_path1 = f'{out_path}/triple_with_wiki_{start}-{end-1}.json'
        out_file_path2 = f'{out_path}/triple_with_wiki_{start}-{end-1}_count.txt'

    results, relation_cnt_freq = make_triple_with_wiki(subject_labels, relation_pids, limit, sparql)
    write_json_to_file(results, out_file_path1)
    write_dict_freq(sorted_dict_key(relation_cnt_freq), out_file_path2)


if __name__ == "__main__":
    home_dir = '/home/nlpshlee/dev_env/git/repos/memit'
    data_dir = f'{home_dir}/data'
    out_path = f'{data_dir}/find_with_wiki_identical_subjects'

    file_names = [
        f'{data_dir}/___org_multi_counterfact.json',
        f'{data_dir}/___org_counterfact.json'
    ]

    make_cf_infos(file_names, out_path)

    # run_250523_make_triple_with_wiki(out_path, 10, 0, 1000)
    # run_250523_make_triple_with_wiki(out_path, 10, 1000, 2000)
    # run_250523_make_triple_with_wiki(out_path, 10, 2000, 3000)
    # run_250523_make_triple_with_wiki(out_path, 10, 3000, 4000)
    # run_250523_make_triple_with_wiki(out_path, 10, 4000, 5000)
    # run_250523_make_triple_with_wiki(out_path, 10, 5000, 6000)
    # run_250523_make_triple_with_wiki(out_path, 10, 6000, 7000)
    # run_250523_make_triple_with_wiki(out_path, 10, 7000, 8000)
    # run_250523_make_triple_with_wiki(out_path, 10, 8000, 9000)
    # run_250523_make_triple_with_wiki(out_path, 10, 9000, 10000)
    # run_250523_make_triple_with_wiki(out_path, 10, 10000, 11000)
    # run_250523_make_triple_with_wiki(out_path, 10, 11000, 12000)
    # run_250523_make_triple_with_wiki(out_path, 10, 12000, 13000)
    # run_250523_make_triple_with_wiki(out_path, 10, 13000, 14000)
    # run_250523_make_triple_with_wiki(out_path, 10, 14000, 15000)
    # run_250523_make_triple_with_wiki(out_path, 10, 15000, 16000)
    # run_250523_make_triple_with_wiki(out_path, 10, 16000, 17000)
    # run_250523_make_triple_with_wiki(out_path, 10, 17000, 18000)
    # run_250523_make_triple_with_wiki(out_path, 10, 18000, 19000)
    # run_250523_make_triple_with_wiki(out_path, 10, 19000, 20000)
    run_250523_make_triple_with_wiki(out_path, 10, 20000, 21000)
    run_250523_make_triple_with_wiki(out_path, 10, 21000, 22000)
    run_250523_make_triple_with_wiki(out_path, 10, 22000, 23000)

