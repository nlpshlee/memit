from .data_preprocessor import *
from .cf_pair import *
from .wiki_util import *

from tqdm import tqdm


SEED = 7
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)

IDX_START = 100000


def _print_and_write(out_path: str, file_name: str, prefix: str, data_set: set=None, data_dict: dict=None):
    if data_set is not None:
        print(f'# wikidata_builder._print_and_write() [{file_name}] {prefix}_set size : {len(data_set)}')
        out_file_path = f'{out_path}/{file_name}/{file_name}_{prefix}_set.txt'
        write_set(data_set, out_file_path)
    if data_dict is not None:
        print(f'# wikidata_builder._print_and_write() [{file_name}] {prefix}_dict size : {len(data_dict)}')
        out_file_path = f'{out_path}/{file_name}/{file_name}_{prefix}_dict.json'
        serializable = { key: sorted(list(value)) for key, value in sorted_dict_key(data_dict).items() }
        write_json_file(serializable, out_file_path)


def _add_relation_object_org_dict(in_dict: dict, relation_id, relation, object_org):
    if relation_id not in in_dict.keys():
        in_dict[relation_id] = {}
    relation_id_dict = in_dict[relation_id]

    if relation not in relation_id_dict.keys():
        relation_id_dict[relation] = {}
    relation_dict = relation_id_dict[relation]

    add_dict_freq(relation_dict, object_org, 1)


def _write_relation_object_org_dict(out_path: str, file_name: str, prefix: str, data_dict: dict):
    print(f'# wikidata_builder._write_relation_object_org_dict() [{file_name}] {prefix}_dict size : {len(data_dict)}')
    out_file_path = f'{out_path}/{file_name}/{file_name}_{prefix}_dict.json'

    for relation_id in data_dict.keys():
        relation_id_dict = data_dict[relation_id]

        for relation in relation_id_dict.keys():
            relation_dict = relation_id_dict[relation]
            relation_dict_sorted = sorted_dict(relation_dict)
            relation_id_dict[relation] = relation_dict_sorted
    
    write_json_file(data_dict, out_file_path)


def make_cf_infos(in_file_paths: List[str], out_path: str):
    subject_set_all = set()
    relation_dict_all = dict()
    object_dict_all = dict()
    object_org_dict_all = dict()
    object_new_dict_all = dict()
    triple_set_all = set()
    relation_object_org_dict_all = dict()

    for in_file_path in in_file_paths:
        file_name = get_file_name(in_file_path, rm_ext_flag=True)

        datas = load_datas(in_file_path)
        subject_set = set()
        relation_dict = dict()
        object_dict = dict()
        object_org_dict = dict()
        object_new_dict = dict()
        triple_set = set()
        relation_object_org_dict = dict()

        for data in datas:
            cf_pair = CFPair(data)

            if cf_pair.validate():            
                subject = cf_pair._subject
                relation_id = cf_pair._relation_id
                relation = cf_pair._prompt
                object_org = cf_pair._target_true_str
                object_org_id = cf_pair._target_true_id
                object_new = cf_pair._target_new_str
                object_new_id = cf_pair._target_new_id

                subject_set.add(subject)
                add_dict_set(relation_dict, relation_id, [relation])
                add_dict_set(object_org_dict, object_org, [object_org_id])
                add_dict_set(object_new_dict, object_new, [object_new_id])
                add_dict_set(object_dict, object_org, [object_org_id])
                add_dict_set(object_dict, object_new, [object_new_id])
                triple_set.add(f'{subject}\t{relation_id}\t{relation}\t{object_org}')
                _add_relation_object_org_dict(relation_object_org_dict, relation_id, relation, object_org)

                subject_set_all.add(subject)
                add_dict_set(relation_dict_all, relation_id, [relation])
                add_dict_set(object_org_dict_all, object_org, [object_org_id])
                add_dict_set(object_new_dict_all, object_new, [object_new_id])
                add_dict_set(object_dict_all, object_org, [object_org_id])
                add_dict_set(object_dict_all, object_new, [object_new_id])
                triple_set_all.add(f'{subject}\t{relation_id}\t{relation}\t{object_org}')
                _add_relation_object_org_dict(relation_object_org_dict_all, relation_id, relation, object_org)
        
        _print_and_write(out_path, file_name, 'subject', data_set=subject_set)
        _print_and_write(out_path, file_name, 'relation', data_dict=relation_dict)
        _print_and_write(out_path, file_name, 'object_org', data_dict=object_org_dict)
        _print_and_write(out_path, file_name, 'object_new', data_dict=object_new_dict)
        _print_and_write(out_path, file_name, 'object', data_dict=object_dict)
        _print_and_write(out_path, file_name, 'triple', data_set=triple_set)
        _write_relation_object_org_dict(out_path, file_name, 'relation_object_org', data_dict=relation_object_org_dict)
    
    _print_and_write(out_path, 'all', 'subject', data_set=subject_set_all)
    _print_and_write(out_path, 'all', 'relation', data_dict=relation_dict_all)
    _print_and_write(out_path, 'all', 'object_org', data_dict=object_org_dict_all)
    _print_and_write(out_path, 'all', 'object_new', data_dict=object_new_dict_all)
    _print_and_write(out_path, 'all', 'object', data_dict=object_dict_all)
    _print_and_write(out_path, 'all', 'triple', data_set=triple_set_all)
    _write_relation_object_org_dict(out_path, 'all', 'relation_object_org', data_dict=relation_object_org_dict_all)


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


def merge_files(in_file_paths: list, out_file_path: str, idx_start=0):
    data_size_all = 0
    subject_set = set()
    
    merged = []
    idx = idx_start

    for in_file_path in in_file_paths:
        datas = load_json_file(in_file_path)
        data_size = len(datas)
        data_size_all += data_size
        print(f'# wikidata_builder.merge_files() in_file_path : {in_file_path}, data_size : {data_size}')

        for data in datas:
            subject = data['subject']
            # subject = remove_space(subject).lower() # 18165 -> 18164
            subject_set.add(subject) # 확인용

            data['subject_idx'] = idx
            idx += 1

            merged.append(data)
    
    write_json_file(merged, out_file_path)
    
    print(f'# wikidata_builder.merge_files() data_size_all : {data_size_all}')
    print(f'# wikidata_builder.merge_files() subject_set size : {len(subject_set)}')
    print(f'# wikidata_builder.merge_files() merged size : {len(merged)}\n')


def _get_relation_object_new(relation_pid_object_dict: dict, relation_pid, object_label):
    relation_object_dict = relation_pid_object_dict[relation_pid]
    relation = get_random_key(relation_object_dict, 1)[0]
    object_label_new = get_random_key(relation_object_dict[relation], 1)[0]
    
    return relation, object_label_new


def add_object_new(datas: list, relation_pid_object_dict: dict, object_map: dict, out_file_path: str):
    for data in datas:
        for relation_dict in data['relations']:
            relation_pid = relation_dict['relation_pid']

            '''
                - WIKI에서 찾을 때는, 하나의 relation_pid에 찾아진 모든 object를 저장은 해놨는데,
                  이걸 개별 데이터로 보면 안될 듯

                    - 일단 relation_pid 별로 여러 relation이 맵핑되어 있는데, 완전 다른 relation으로 보기 보다는
                      의미가 유사한, paraphrase 또는 generation 느낌이라
                      개별 적으로 분리해서 사용하면 안 될 것 같고, 첫 번째 object만 사용
            '''
            # for object in relation_dict['objects']:
            #     object_label = object['object_label']

            object_dict = relation_dict['objects'][0]
            object_label = object_dict['object_label']
            
            relation, object_label_new = _get_relation_object_new(relation_pid_object_dict, relation_pid, object_label)
            object_pid_new = object_map[object_label_new][0]

            relation_dict['relation'] = relation
            object_dict['object_label_new'] = object_label_new
            object_dict['object_pid_new'] = object_pid_new

    # relation(prompt)와 편집하려는 object까지 추가된 상태
    write_json_file(datas, out_file_path)


def convert_to_mcf(datas: list, out_file_path: str, idx_start=0):
    result = []
    idx = idx_start
    pararel_idx = -1

    for data in datas:
        subject = data['subject']

        for relation_dict in data['relations']:
            prompt = relation_dict['relation']
            relation_id = relation_dict['relation_pid']

            object_dict = relation_dict['objects'][0]
            target_new = {'str': object_dict['object_label_new'], 'id': object_dict['object_pid_new']}
            target_true = {'str': object_dict['object_label'], 'id': object_dict['object_pid']}

            data_new = {}
            data_new['case_id'] = idx
            idx += 1

            data_new['pararel_idx'] = pararel_idx
            
            requested_rewrite = {
                'prompt': prompt,
                'relation_id': relation_id,
                'target_new': target_new,
                'target_true': target_true,
                'subject': subject
            }
            data_new['requested_rewrite'] = requested_rewrite

            query = prompt.format(subject)
            data_new['paraphrase_prompts'] = [query]
            data_new['neighborhood_prompts'] = [query]
            data_new['attribute_prompts'] = [query]
            data_new['generation_prompts'] = [query]

            result.append(data_new)
    
    write_json_file(result, out_file_path)


def run_250523_make_triple_with_wiki(out_path: str, limit, start=-1, end=-1):
    sparql = get_sparql()

    subject_set = set()
    in_file_path = f'{out_path}/all/all_subject_set.txt'
    load_set(in_file_path, subject_set)
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
        end = min(end, len(subject_labels))-1
        print(f'start : {start}, end : {end}\n')
        subject_labels = subject_labels[start:(end+1)]
        out_file_path1 = f'{out_path}/maked_json/triple_with_wiki_{start}-{end}.json'
        out_file_path2 = f'{out_path}/maked_count/triple_with_wiki_{start}-{end}_count.txt'

    results, relation_cnt_freq = make_triple_with_wiki(subject_labels, relation_pids, limit, sparql)
    write_json_file(results, out_file_path1)
    write_dict_freq(sorted_dict_key(relation_cnt_freq), out_file_path2)


def run_250523_make_triple_with_wiki_all(out_path: str):
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


def run_250531_merge_files(in_path: str):
    in_file_paths = sorted(get_file_paths(f'{in_path}/maked_json'))
    out_file_path = f'{in_path}/1. find_with_wiki_identical_subjects_merged.json'
    merge_files(in_file_paths, out_file_path, IDX_START)

    in_file_paths = sorted(get_file_paths(f'{in_path}/maked_count'))
    count_dict = {}

    for in_file_path in in_file_paths:
        load_freq_dict(in_file_path, count_dict)
    
    out_file_path = f'{in_path}/1. find_with_wiki_identical_subjects_merged_count.txt'
    write_dict_freq(count_dict, out_file_path)


def run_250531_add_object_new(in_path: str):
    in_file_path = f'{in_path}/1. find_with_wiki_identical_subjects_merged.json'
    datas = load_datas(in_file_path)

    in_file_path = f'{in_path}/all/all_relation_object_org_dict.json'
    relation_pid_object_dict = load_datas(in_file_path)

    in_file_path = f'{in_path}/all/all_object_dict.json'
    object_map = load_datas(in_file_path)

    out_file_path = f'{in_path}/2. find_with_wiki_identical_subjects_add_object_new.json'
    add_object_new(datas, relation_pid_object_dict, object_map, out_file_path)


def run_250531_convert_to_mcf(in_path: str):
    in_file_path = f'{in_path}/2. find_with_wiki_identical_subjects_add_object_new.json'
    datas = load_datas(in_file_path)

    out_file_path = f'{in_path}/3. find_with_wiki_identical_subjects_convert_mcf.json'
    convert_to_mcf(datas, out_file_path, IDX_START)


if __name__ == "__main__":
    home_dir = '/home/nlpshlee/dev_env/git/repos/memit'
    data_dir = f'{home_dir}/data'
    out_path = f'{data_dir}/find_with_wiki_identical_subjects'

    file_names = [
        f'{data_dir}/___org_multi_counterfact.json',
        f'{data_dir}/___org_counterfact.json'
    ]

    make_cf_infos(file_names, out_path)
    # run_250523_make_triple_with_wiki_all(out_path)
    run_250531_merge_files(out_path)
    run_250531_add_object_new(out_path)
    run_250531_convert_to_mcf(out_path)

