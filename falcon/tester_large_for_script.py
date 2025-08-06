import argparse

from .falcon_util import *
from .model_editor import *
from .tester import load_datas, get_model_editor


SEED = 7
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

'''
    다음 에러에 대한 해결 방안
        RuntimeError: cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR, when calling `cusolverDnCreate(handle)`.
        If you keep seeing this error, you may use `torch.backends.cuda.preferred_linalg_library()` to try linear algebra operators with other supported backends.
        See https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.preferred_linalg_library
    
    버전 문제로
        cuSOLVER 대신 magma
            일부 상황에서만 에러가 발생하긴 함
            (GPT-J, TWO-STEP)
'''
torch.backends.cuda.preferred_linalg_library("magma")


def get_in_file_path_subject_mul(data_dir, identical_num, num_edits, identical_ratio):
    in_path = f'{data_dir}/multiple_identical_subjects'
    file_name = 'mcf_multiple_identical{}_subjects_{}_{}:{}{}.json'
    file_name = file_name.format(identical_num, num_edits, 10-identical_ratio, identical_ratio, '')
    in_file_path = f'{in_path}/identical{identical_num}/{file_name}'

    return in_file_path


def get_in_file_path_relation_mul(data_dir, identical_num, num_edits, identical_ratio):
    in_path = f'{data_dir}/multiple_identical_subjects'
    file_name = 'mcf_multiple_identical{}_subjects_{}_{}:{}{}.json'
    file_name = file_name.format(identical_num, num_edits, 10-identical_ratio, identical_ratio, '_sr_swap_post')
    in_file_path = f'{in_path}/identical{identical_num}/{file_name}'

    return in_file_path


def get_in_file_path_subject_seq(data_dir, identical_num, batch_idx):
    in_path = f'{data_dir}/sequential_identical_subjects/each'
    file_name = 'mcf_sequential_identical{}_subjects_batch{}{}.json'
    in_file_path = f'{in_path}/identical{identical_num}/' + file_name.format(identical_num, batch_idx, '')

    return in_file_path


def get_in_file_path_relation_seq(data_dir, identical_num, batch_idx):
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





def run_memit_mul(alg_name: str, model_name: str, data_dir: str, identical_num: int, num_edits: int, identical_ratio: int,
                  layers_subject: list, mode: str, model_dir: str):

    in_file_path = get_in_file_path_subject_mul(data_dir, identical_num, num_edits, identical_ratio)
    datas_subject = load_datas(in_file_path)

    model_editor = get_model_editor(num_edits, alg_name=alg_name, model_name=model_name, do_init_model=True)

    if data_dir.endswith('org'):
        model_editor._do_eval_new_model = True

    # 모델 편집
    model_editor.set_params_external({'layers': layers_subject})
    model_editor.edit_ext_datas(datas_subject, False, True, True, False, False, False)

    # 모델 저장
    model_path = get_model_path(model_dir, model_name, identical_num, num_edits, mode)
    model_editor.model_save(model_path)


def run_two_step_mul(alg_name: str, model_name: str, data_dir: str, identical_num: int, num_edits: int, identical_ratio: int,
                     layers_subject: list, layers_relation: list, mode: str, model_dir: str):

    model_path = get_model_path(model_dir, model_name, identical_num, num_edits, mode)

    if mode.endswith('SUBJECT'):
        in_file_path = get_in_file_path_subject_mul(data_dir, identical_num, num_edits, identical_ratio)

        model_editor = get_model_editor(num_edits, alg_name=alg_name, model_name=model_name, do_init_model=True)
        model_editor.set_params_external({'layers': layers_subject})
        do_edit_test = False

    elif mode.endswith('RELATION'):
        in_file_path = get_in_file_path_relation_mul(data_dir, identical_num, num_edits, identical_ratio)

        model_editor = get_model_editor(num_edits, alg_name=alg_name, model_name=model_name, do_init_model=False)
        model_editor.model_load(model_path)
        model_editor._model.config._name_or_path = model_name
        model_editor.set_params_external({'layers': layers_relation})
        do_edit_test = True

        if data_dir.endswith('org'):
            model_editor._do_eval_new_model = True
    
    # 모델 편집 및 저장
    datas = load_datas(in_file_path)
    model_editor.edit_ext_datas(datas, False, True, do_edit_test, False, False, False)
    model_editor.model_save(model_path)


def run_memit_seq(alg_name: str, model_name: str, data_dir: str, identical_num: int, num_edits: int, batch_idx: int,
                  layers_subject: list, mode: str, model_dir: str):

    in_file_path = get_in_file_path_subject_seq(data_dir, identical_num, batch_idx)
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
    if identical_num == batch_idx and data_dir.endswith('org'):
        model_editor._do_eval_org_model = True

    # 현재까지의 모든 배치 단위로 성능 측정
    for batch_idx_cur in range(1, batch_idx+1):
        print(f'# falcon.tester_large_for_script.run_memit() batch_idx : {batch_idx}, batch_idx_cur : {batch_idx_cur}\n')

        in_file_path = get_in_file_path_subject_seq(data_dir, identical_num, batch_idx_cur)
        datas_batch = load_datas(in_file_path)
        model_editor.edit_ext_datas(datas_batch, True, False, False, False, False, False)


'''
    [SUBJECT, RELATION] 둘 중에 하나만 실행되도록 설계
'''
def run_two_step_seq(alg_name: str, model_name: str, data_dir: str, identical_num: int, num_edits: int, batch_idx: int,
                     layers_subject: list, layers_relation: list, mode: str, model_dir: str):

    model_path = get_model_path(model_dir, model_name, identical_num, num_edits, mode)

    if mode.endswith('SUBJECT'):
        in_file_path = get_in_file_path_subject_seq(data_dir, identical_num, batch_idx)
        
        if batch_idx == 1:
            model_editor = get_model_editor(num_edits, alg_name=alg_name, model_name=model_name, do_init_model=True)
        else:
            model_editor = get_model_editor(num_edits, alg_name=alg_name, model_name=model_name, do_init_model=False)
            model_editor.model_load(model_path)
            model_editor._model.config._name_or_path = model_name
        
        model_editor.set_params_external({'layers': layers_subject})

    elif mode.endswith('RELATION'):
        in_file_path = get_in_file_path_relation_seq(data_dir, identical_num, batch_idx)

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
        if identical_num == batch_idx and data_dir.endswith('org'):
            model_editor._do_eval_org_model = True
        
        # 현재까지의 모든 배치 단위로 성능 측정
        for batch_idx_cur in range(1, batch_idx+1):
            print(f'# falcon.tester_large_for_script.run_two_step() batch_idx : {batch_idx}, batch_idx_cur : {batch_idx_cur}\n')

            in_file_path = get_in_file_path_subject_seq(data_dir, identical_num, batch_idx_cur)
            datas_batch = load_datas(in_file_path)
            model_editor.edit_ext_datas(datas_batch, True, False, False, False, False, False)


def run(alg_name: str, model_name: str, data_dir, identical_num: int, num_edits: int,
        identical_ratio: int, batch_idx: int, mode: str, model_dir: str):

    print(f'# falcon.tester_large_for_script.run()')
    print(f'\talg_name : {alg_name}')
    print(f'\tmodel_name : {model_name}')
    print(f'\tdata_dir : {data_dir}')
    print(f'\tidentical_num : {identical_num}')
    print(f'\tnum_edits : {num_edits}')
    print(f'\tidentical_ratio : {identical_ratio}')
    print(f'\tbatch_idx : {batch_idx}')
    print(f'\tmode : {mode}')
    print(f'\tmodel_dir : {model_dir}\n')

    layers_subject, layers_relation = get_edit_layers(model_name)

    # Multiple Test
    if identical_ratio != -1 and batch_idx == -1:
        if mode == 'MEMIT':
            run_memit_mul(alg_name, model_name, data_dir, identical_num, num_edits, identical_ratio,
                          layers_subject, mode, model_dir)
        elif mode.startswith('TWO-STEP'):
            run_two_step_mul(alg_name, model_name, data_dir, identical_num, num_edits, identical_ratio,
                             layers_subject, layers_relation, mode, model_dir)

    # Sequential Test
    elif identical_ratio == -1 and batch_idx != -1:
        if mode == 'MEMIT':
            run_memit_seq(alg_name, model_name, data_dir, identical_num, num_edits, batch_idx,
                    layers_subject, mode, model_dir)

        elif mode.startswith('TWO-STEP'):
            run_two_step_seq(alg_name, model_name, data_dir, identical_num, num_edits, batch_idx,
                        layers_subject, layers_relation, mode, model_dir)


def run_test_for_script():
    data_dir = './data/preprocessing_org'
    model_dir = './models'

    alg_name = 'MEMIT'
    model_name = 'gpt2-xl'

    identical_nums = [2, 3, 4]
    num_edits_list = [500, 35, 5]
    modes = ['MEMIT', 'TWO-STEP_SUBJECT', 'TWO-STEP_RELATION']

    for identical_num, num_edits in zip(identical_nums, num_edits_list):
        for batch_idx in range(1, identical_num+1):
            for mode in modes:
                run(alg_name, model_name, data_dir, identical_num, num_edits, batch_idx, mode, model_dir)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg_name', type=str, required=True, help="Algorithm name")
    parser.add_argument('--model_name', type=str, required=True, help="Model name")
    parser.add_argument('--data_dir', type=str, required=True, help="Data dir")
    parser.add_argument('--identical_num', type=int, required=True, help="Number of identicals")
    parser.add_argument('--num_edits', type=int, required=True, help="Number of edits")
    parser.add_argument('--identical_ratio', type=int, required=False, default=-1, help="Identical ratio")
    parser.add_argument('--batch_idx', type=int, required=False, default=-1, help="Batch index")
    parser.add_argument('--mode', type=str, required=True, help="Mode flag")
    parser.add_argument('--model_dir', type=str, required=True, help="Model dir")
    args = parser.parse_args()

    alg_name = args.alg_name
    model_name = args.model_name
    data_dir = args.data_dir
    identical_num = args.identical_num
    num_edits = args.num_edits
    identical_ratio = args.identical_ratio
    batch_idx = args.batch_idx
    mode = args.mode
    model_dir = args.model_dir

    if model_name == 'gpt-j':
        model_name = 'EleutherAI/gpt-j-6B'

    run(alg_name, model_name, data_dir, identical_num, num_edits,
        identical_ratio, batch_idx, mode, model_dir)
    
    '''
        매 편집 스텝마다, 모델을 저장하고 불러오는 것 자체는 문제가 없음
        하지만, 한 번의 파이썬 실행에서 여러 편집 스텝을 연속으로 처리 하는 것과
        하나의 파이썬 실행에서 한 번의 편집만을 수행하고 이를 여러 파이썬 호출로 처리하면
        매번 파이썬이 다시 실행되면서 동일한 시드를 줘도 난수값이 달라지거나, 그 외 다른 이유로 결과가 달라질 수 있음
        완전 동일한 코드를 스크립트로 여러번 호출하는 것과 하나의 함수로 묶어서 딱 한 번 실행하는 것의 결과가 다름
        하나의 함수로 딱 한 번 실행하는건 원래 성능과 동일함을 확인
        즉, 코드 레벨에서는 문제가 없고 파이썬을 여러번 호출하는 것 자체가 원인
    '''
    # run_test_for_script()

