import argparse
from copy import deepcopy
from tqdm import tqdm

from .falcon_util import *
from .model_editor import *


SEED = 7
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def load_datas(in_file_path: str):
    in_file = open_file(in_file_path, mode='r')
    datas = json.load(in_file)

    print(f'# falcon.tester_only_seq_large.load_datas() datas size : {len(datas)}, in_file_path : {in_file_path}')
    return datas


def get_model_editor(num_edits=100, hparams_fname_suffix='', hparams_mod=None, alg_name='MEMIT', model_name='gpt2-xl',
                     do_init_model=True):

    hparams_fname = model_name + '{}.json'.format(hparams_fname_suffix)
    ds_name = 'mcf'

    dataset_size_limit = None
    continue_from_run = None
    skip_generation_tests = False
    generation_test_interval = 1
    conserve_memory = False
    dir_name = alg_name
    use_cache = False
    output_hidden_states = False

    model_editor = ModelEditor(
        alg_name, model_name, hparams_fname, ds_name,
        dataset_size_limit, continue_from_run, skip_generation_tests,
        generation_test_interval, conserve_memory, dir_name, num_edits, use_cache, output_hidden_states,
        hparams_mod, do_init_model
    )

    return model_editor


def copy_weights(model_editor: ModelEditor, layers_subject: list, layers_relation: list):
    layers_all = deepcopy(layers_subject)
    layers_all.extend(layers_relation)
    print(f'### falcon.tester.copy_weights() layers_subject : {layers_subject}')
    print(f'### falcon.tester.copy_weights() layers_relation : {layers_relation}')
    print(f'### falcon.tester.copy_weights() layers_all : {layers_all}\n')

    return model_editor.copy_weights(layers_all)


def run_multiple(data_dir: str, identical_nums: list, num_edits: int, mode: int,
                 alg_name='MEMIT', model_name='gpt2-xl', layers_subject=[13, 14, 15, 16, 17], layers_relation=[26, 27, 28, 29, 30]):

    in_path = f'{data_dir}/multiple_identical_subjects'
    file_name = 'mcf_multiple_identical{}_subjects_{}_{}:{}{}.json'

    model_editor = get_model_editor(num_edits, alg_name=alg_name, model_name=model_name)
    weights_copy = copy_weights(model_editor, layers_subject, layers_relation)

    for identical_num in identical_nums:
        for i in tqdm([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]):
            in_file_path = f'{in_path}/identical{identical_num}/' + file_name.format(identical_num, num_edits, i, (10-i), '')
            datas_subject = load_datas(in_file_path)

            model_editor._do_eval_new_model = False
            do_edit_test = False
            if check_option(mode, MODE.ORG):
                model_editor._do_eval_new_model = True
                do_edit_test = True
            
            # 1. 기존 Subject 마지막 토큰으로 모델 편집
            model_editor.set_params_external({'layers': layers_subject})
            model_editor.edit_ext_datas(datas_subject, False, True, do_edit_test, False, False, False)

            # 2. 제안 방법 Subject-Relation Two-Step 모델 편집
            if check_option(mode, MODE.TWO_STEP):
                in_file_path = f'{in_path}/identical{identical_num}/' + file_name.format(identical_num, num_edits, i, (10-i), '_sr_swap_post')
                datas_relation = load_datas(in_file_path)

                model_editor._do_eval_new_model = True
                model_editor.set_params_external({'layers': layers_relation})
                model_editor.edit_ext_datas(datas_relation, False, True, True, False, False, False)
            
            # 3. 가중치 복원 및 결과 폴더 재설정
            model_editor.restore_weights(weights_copy)
            model_editor.check_continue_from_run()


def run_sequential(data_dir: str, identical_nums: list, num_edits_list: list, mode: int,
                   alg_name='MEMIT', model_name='gpt2-xl', layers_subject=[13, 14, 15, 16, 17], layers_relation=[26, 27, 28, 29, 30]):

    in_path = f'{data_dir}/sequential_identical_subjects/each'
    file_name = 'mcf_sequential_identical{}_subjects_batch{}{}.json'

    model_editor = get_model_editor(alg_name=alg_name, model_name=model_name)
    weights_copy = copy_weights(model_editor, layers_subject, layers_relation)

    for identical_num, num_edits in zip(identical_nums, num_edits_list):
        model_editor._num_edits = num_edits
        model_editor._do_eval_org_model = False
        model_editor._print_init()

        datas_batchs = []

        for batch_idx in tqdm(range(1, identical_num+1)):
            print(f'### falcon.tester.run_sequential() identical : {identical_num}, batch_idx : {batch_idx}, batch_size : {num_edits}\n')

            in_file_path = f'{in_path}/identical{identical_num}/' + file_name.format(identical_num, batch_idx, '')
            datas_subject = load_datas(in_file_path)

            # 1. 기존 Subject 마지막 토큰으로 모델 편집
            model_editor.set_params_external({'layers': layers_subject})
            model_editor.edit_ext_datas(datas_subject, False, True, False, False, False, False)

            # 2. 제안 방법 Subject-Relation Two-Step 모델 편집
            if check_option(mode, MODE.TWO_STEP):
                in_file_path = f'{in_path}/identical{identical_num}/' + file_name.format(identical_num, batch_idx, '_sr_swap_post')
                datas_relation = load_datas(in_file_path)

                model_editor.set_params_external({'layers': layers_relation})
                model_editor.edit_ext_datas(datas_relation, False, True, False, False, False, False)
            
            # 3. 배치 단위 성능 측정
            datas_batchs.append(datas_subject)
            print(f'\n### falcon.tester.run_sequential() batch size : {len(datas_batchs)}\n')

            # 마지막 배치 스텝에서 원래 논문 성능 평가를 위한 실험 진행
            if batch_idx == identical_num:
                model_editor._do_eval_org_model = True

            for i, datas_batch in enumerate(datas_batchs):
                print(f'### falcon.tester.run_sequential() batch_{i+1} data size : {len(datas_batch)}\n')
                model_editor.edit_ext_datas(datas_batch, True, False, False, False, False, False)
        
        # 4. 가중치 복원 및 결과 폴더 재설정
        model_editor.restore_weights(weights_copy)
        model_editor.check_continue_from_run()





def run_250723_save_and_load(do_save=False, do_load=False):
    home_dir = '/home/nlpshlee/dev_env/git/repos/memit'
    data_dir = f'{home_dir}/data/preprocessing_org'
    model_dir = f'{home_dir}/models'

    identical_nums = [3]
    num_edits_list = [35]
    alg_name = 'MEMIT'
    model_name = 'gpt2-xl'
    layers_subject = [13, 14, 15, 16, 17]
    # layers_relation = [26, 27, 28, 29, 30]

    in_path = f'{data_dir}/sequential_identical_subjects/each'
    file_name = 'mcf_sequential_identical{}_subjects_batch{}{}.json'

    model_editor = get_model_editor(alg_name=alg_name, model_name=model_name, do_init_model=(not do_load))
    check_gpu_memory()

    for identical_num, num_edits in zip(identical_nums, num_edits_list):
        model_editor._num_edits = num_edits
        model_editor._print_init()

        for batch_idx in tqdm(range(1, identical_num+1)):
            print(f'### falcon.tester.run_250723_save_and_load() identical : {identical_num}, batch_idx : {batch_idx}, batch_size : {num_edits}\n')

            in_file_path = f'{in_path}/identical{identical_num}/' + file_name.format(identical_num, batch_idx, '')
            datas_subject = load_datas(in_file_path)

            save_path = f'{model_dir}/identical{identical_num}_{identical_num*num_edits}_batch{batch_idx}'

            if do_save:
                model_editor.set_params_external({'layers': layers_subject})
                model_editor.edit_ext_datas(datas_subject, False, True, True, False, False, False)
                model_editor.model_save(save_path)
            if do_load:
                model_editor.model_load(save_path)
                check_gpu_memory()
                model_editor.edit_ext_datas(datas_subject, True, False, False, False, False, False)
            break
        break


def get_in_file_path_subject(data_dir, identical_num, batch_idx):
    in_path = f'{data_dir}/sequential_identical_subjects/each'
    file_name = 'mcf_sequential_identical{}_subjects_batch{}{}.json'
    in_file_path = f'{in_path}/identical{identical_num}/' + file_name.format(identical_num, batch_idx, '')
    return in_file_path


def get_model_path(model_dir, model_name, identical_num, num_edits, batch_idx, mode):
    model_path = f'{model_dir}/{model_name}_identical{identical_num}_{identical_num*num_edits}_batch{batch_idx}_{mode}'
    return model_path


def run_memit(alg_name: str, model_name: str, data_dir, identical_num: int, num_edits: int, batch_idx: int, layers_subject: list, mode: str, model_dir: str):
    in_file_path = get_in_file_path_subject(data_dir, identical_num, batch_idx)
    datas_subject = load_datas(in_file_path)

    if batch_idx == 1:
        model_editor = get_model_editor(num_edits, alg_name=alg_name, model_name=model_name, do_init_model=True)
    else:
        model_editor = get_model_editor(num_edits, alg_name=alg_name, model_name=model_name, do_init_model=False)
        model_path = get_model_path(model_dir, model_name, identical_num, num_edits, batch_idx-1, mode)
        model_editor.model_load(model_path)
    
    model_editor.set_params_external({'layers': layers_subject})
    model_editor.edit_ext_datas(datas_subject, False, True, True, False, False, False)

    model_path = get_model_path(model_dir, model_name, identical_num, num_edits, batch_idx, mode)
    model_editor.model_save(model_path)


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

    if mode == 'MEMIT':
        run_memit(alg_name, model_name, data_dir, identical_num, num_edits, batch_idx, [13, 14, 15, 16, 17], mode, model_dir)
    elif mode == 'TWO-STEP_SUBJECT':
        pass
    elif mode == 'TWO-STEP_RELATION':
        pass


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

