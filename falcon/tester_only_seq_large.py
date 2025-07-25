import argparse

from .falcon_util import *
from .model_editor import *
from .tester import load_datas, get_model_editor


SEED = 7
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def get_in_file_path_subject(data_dir, identical_num, batch_idx):
    in_path = f'{data_dir}/sequential_identical_subjects/each'
    file_name = 'mcf_sequential_identical{}_subjects_batch{}{}.json'
    in_file_path = f'{in_path}/identical{identical_num}/' + file_name.format(identical_num, batch_idx, '')

    return in_file_path


def get_in_file_path_relation(data_dir, identical_num, batch_idx):
    in_path = f'{data_dir}/sequential_identical_subjects/each'
    file_name = 'mcf_sequential_identical{}_subjects_batch{}{}.json'
    in_file_path = f'{in_path}/identical{identical_num}/' + file_name.format(identical_num, batch_idx, '_sr_swap_post')

    return in_file_path


def get_model_path(model_dir, model_name: str, identical_num, num_edits, mode: str):
    if '/' in model_name:
        model_name = model_name.split('/')[-1]

    if '_' in mode:
        mode = mode.split('_')[0]

    model_path = f'{model_dir}/{model_name}_identical{identical_num}_{identical_num*num_edits}_{mode}'

    return model_path


def get_edit_layers(model_name: str):
    if model_name == 'gpt2-xl':
        layers_subject = [13, 14, 15, 16, 17]
        layers_relation = [26, 27, 28, 29, 30]
    elif 'gpt-j' in model_name:
        layers_subject = [3, 4, 5, 6, 7, 8]
        layers_relation = [11, 12, 13, 14, 15, 16]
    
    return layers_subject, layers_relation





def run_memit(alg_name: str, model_name: str, data_dir, identical_num: int, num_edits: int, batch_idx: int,
              layers_subject: list, mode: str, model_dir: str):

    in_file_path = get_in_file_path_subject(data_dir, identical_num, batch_idx)
    datas_subject = load_datas(in_file_path)

    model_path = get_model_path(model_dir, model_name, identical_num, num_edits, mode)

    if batch_idx == 1:
        model_editor = get_model_editor(num_edits, alg_name=alg_name, model_name=model_name, do_init_model=True)
    else:
        model_editor = get_model_editor(num_edits, alg_name=alg_name, model_name=model_name, do_init_model=False)
        model_editor.model_load(model_path)
        # cov key 값을 가져오려면, 실제 모델 객체에 모델 원본명을 설정해야 함
        model_editor._model.config._name_or_path = model_name
    
    # 모델 편집 및 저장
    model_editor.set_params_external({'layers': layers_subject})
    model_editor.edit_ext_datas(datas_subject, False, True, False, False, False, False)
    model_editor.model_save(model_path)

    # 마지막 배치 스텝에서 원래 논문 성능 평가를 위한 실험 진행
    if identical_num == batch_idx:
        model_editor._do_eval_org_model = True

    # 현재까지의 모든 배치 단위로 성능 측정
    for batch_idx_cur in range(1, batch_idx+1):
        print(f'# falcon.tester_only_seq_large.run_memit() batch_idx : {batch_idx}, batch_idx_cur : {batch_idx_cur}\n')

        in_file_path = get_in_file_path_subject(data_dir, identical_num, batch_idx_cur)
        datas_batch = load_datas(in_file_path)
        model_editor.edit_ext_datas(datas_batch, True, False, False, False, False, False)


'''
    [SUBJECT, RELATION] 둘 중에 하나만 실행되도록 설계
'''
def run_two_step(alg_name: str, model_name: str, data_dir, identical_num: int, num_edits: int, batch_idx: int,
                 layers_subject: list, layers_relation: list, mode: str, model_dir: str):

    model_path = get_model_path(model_dir, model_name, identical_num, num_edits, mode)

    if mode.endswith('SUBJECT'):
        in_file_path = get_in_file_path_subject(data_dir, identical_num, batch_idx)
        
        if batch_idx == 1:
            model_editor = get_model_editor(num_edits, alg_name=alg_name, model_name=model_name, do_init_model=True)
        else:
            model_editor = get_model_editor(num_edits, alg_name=alg_name, model_name=model_name, do_init_model=False)
            model_editor.model_load(model_path)
            model_editor._model.config._name_or_path = model_name
        
        model_editor.set_params_external({'layers': layers_subject})

    elif mode.endswith('RELATION'):
        in_file_path = get_in_file_path_relation(data_dir, identical_num, batch_idx)

        model_editor = get_model_editor(num_edits, alg_name=alg_name, model_name=model_name, do_init_model=False)
        model_editor.model_load(model_path)
        model_editor._model.config._name_or_path = model_name

        model_editor.set_params_external({'layers': layers_relation})
    
    # 모델 편집 및 저장
    datas = load_datas(in_file_path)
    model_editor.edit_ext_datas(datas, False, True, False, False, False, False)
    model_editor.model_save(model_path)

    if mode.endswith('RELATION'):
        # 마지막 배치 스텝에서 원래 논문 성능 평가를 위한 실험 진행
        if identical_num == batch_idx:
            model_editor._do_eval_org_model = True
        
        # 현재까지의 모든 배치 단위로 성능 측정
        for batch_idx_cur in range(1, batch_idx+1):
            print(f'# falcon.tester_only_seq_large.run_two_step() batch_idx : {batch_idx}, batch_idx_cur : {batch_idx_cur}\n')

            in_file_path = get_in_file_path_subject(data_dir, identical_num, batch_idx_cur)
            datas_batch = load_datas(in_file_path)
            model_editor.edit_ext_datas(datas_batch, True, False, False, False, False, False)


def run(alg_name: str, model_name: str, data_dir, identical_num: int, num_edits: int, batch_idx: int, mode: str, model_dir: str):
    print(f'# falcon.tester_only_seq_large.run()')
    print(f'\talg_name : {alg_name}')
    print(f'\tmodel_name : {model_name}')
    print(f'\tdata_dir : {data_dir}')
    print(f'\tidentical_num : {identical_num}')
    print(f'\tnum_edits : {num_edits}')
    print(f'\tbatch_idx : {batch_idx}')
    print(f'\tmode : {mode}')
    print(f'\tmodel_dir : {model_dir}\n')

    layers_subject, layers_relation = get_edit_layers(model_name)

    if mode == 'MEMIT':
        run_memit(alg_name, model_name, data_dir, identical_num, num_edits, batch_idx,
                  layers_subject, mode, model_dir)

    elif mode.startswith('TWO-STEP'):
        run_two_step(alg_name, model_name, data_dir, identical_num, num_edits, batch_idx,
                     layers_subject, layers_relation, mode, model_dir)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg_name', type=str, required=True, help="Algorithm name")
    parser.add_argument('--model_name', type=str, required=True, help="Model name")
    parser.add_argument('--data_dir', type=str, required=True, help="Data dir")
    parser.add_argument('--identical_num', type=int, required=True, help="Number of identicals")
    parser.add_argument('--num_edits', type=int, required=True, help="Number of edits")
    parser.add_argument('--batch_idx', type=int, required=True, help="Batch index")
    parser.add_argument('--mode', type=str, required=True, help="Mode flag")
    parser.add_argument('--model_dir', type=str, required=True, help="Model dir")
    args = parser.parse_args()

    alg_name = args.alg_name
    model_name = args.model_name
    data_dir = args.data_dir
    identical_num = args.identical_num
    num_edits = args.num_edits
    batch_idx = args.batch_idx
    mode = args.mode
    model_dir = args.model_dir

    if model_name == 'gpt-j':
        model_name = 'EleutherAI/gpt-j-6B'

    run(alg_name, model_name, data_dir, identical_num, num_edits, batch_idx, mode, model_dir)

