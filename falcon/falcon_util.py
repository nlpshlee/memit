import os, sys, json, string, re, gc, random
from itertools import islice

import torch

import warnings
warnings.filterwarnings('ignore')


ENCODING = "UTF-8"
DELIM_KEY = "\t"
FILE_EXT = "."


# 모든 일반적인 기호 포함
PUNCTUATION = string.punctuation


class LOG_OPTION:
    STDOUT = 1
    STDERR = 2


def check_option(option1: int, option2: int):
    if option1 == option2:
        return True
    elif (option1 & option2) != 0:
        return True
    else:
        return False


def logging(msg: str, option=LOG_OPTION.STDOUT):
    if check_option(option, LOG_OPTION.STDOUT):
        print(msg)
    if check_option(option, LOG_OPTION.STDERR):
        print(msg, file=sys.stderr)


def logging_error(call_path: str, e: Exception):
    logging(f"### (ERROR) {call_path} error : {e}\n", LOG_OPTION.STDERR)


def is_empty(text: str, trim_flag=True):
    if text is None:
        return True
    
    if trim_flag:
        text = text.strip()
    
    if len(text) == 0:
        return True
    
    return False


def is_symbol(text: str, symbols=PUNCTUATION):
    if is_empty(text):
        return False

    for c in text:
        if c == ' ' or c == '\t' or c == '\n':
            continue
        if not c in symbols:
            return False

    return True


def contains_symbol(text: str, symbols=PUNCTUATION):
    if is_empty(text):
        return False

    for c in text:
        if c in symbols:
            return True

    return False


def remove_space(text: str):
    return re.sub(r'[ \t\n]+', '', text)


def remove_delim_multi(text: str):
    return re.sub(r'[\t\n ]+', ' ', text).strip()


def remove_symbol_edge(text: str, symbols=PUNCTUATION):
    return text.strip(symbols)


def get_file_name(file_path: str, rm_ext_flag=False):
    file_name = os.path.basename(file_path)

    if rm_ext_flag:
        idx = file_name.rfind(FILE_EXT)

        if idx != -1:
            file_name = file_name[:idx]
    
    return file_name


def get_file_paths(in_path: str, inner_flag=True):
    file_paths = []

    if inner_flag:
        for (parent_path, dirs, file_names) in os.walk(in_path):
                for file_name in file_names:
                    file_path = os.path.join(parent_path, file_name)

                    if os.path.isfile(file_path):
                        file_paths.append(file_path)
    else:
        file_names = os.listdir(in_path)
        for file_name in file_names:
            file_path = os.path.join(in_path, file_name)

            if os.path.isfile(file_path):
                file_paths.append(file_path)

    return file_paths


def exists(file_path: str):
    if file_path == None or len(file_path) == 0:
        return False

    if os.path.exists(file_path) and os.path.isfile(file_path):
        return True

    return False


def make_parent(file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def open_file(file_path: str, encoding=ENCODING, mode='r'):
    if mode.find('w') != -1 or mode.find('a') != -1:
        make_parent(file_path)

    if is_empty(encoding, True) or mode.find('b') != -1:
        return open(file_path, mode)
    else:
        return open(file_path, mode, encoding=encoding)


def load_freq_dict(in_file_path: str, in_dict: dict, encoding=ENCODING, delim_key=DELIM_KEY, do_print=False):
    in_file = open_file(in_file_path, encoding, mode='r')

    while 1:
        line = in_file.readline()
        if not line:
            break

        temp = line.strip().split(delim_key)
        key = temp[0].strip()
        value = int(temp[1].strip())

        add_dict_freq(in_dict, key, value)
    in_file.close()

    if do_print:
        logging(f'# falcon_util.load_freq_dict() size : {len(in_dict)}')
    return in_dict


def load_set(in_file_path: str, in_set: set, encoding=ENCODING, do_print=False):
    in_file = open_file(in_file_path, encoding, mode='r')

    while 1:
        line = in_file.readline()
        if not line:
            break

        in_set.add(line.strip())
    in_file.close()
    
    if do_print:
        logging(f'# falcon_util.load_set() size : {len(in_set)}')
    return in_set


def write_dict_freq(out_dict: dict, out_file_path: str, encoding=ENCODING, delim=DELIM_KEY):
    file = open_file(out_file_path, encoding, 'w')

    items = out_dict.items()
    for item in items:
        file.write(f"{item[0]}{delim}{item[1]}\n")
    
    file.close()


def write_set(out_set: set, out_file_path: str, encoding=ENCODING, is_reverse=False):
    file = open_file(out_file_path, encoding, 'w')

    out_list = list(out_set)
    out_list.sort(reverse = is_reverse)

    for i in range(len(out_list)):
        file.write(f"{out_list[i]}\n")
    
    file.close()


def window(seq, n=2):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    for i in range(0, len(arr), n):
        yield arr[i: i + n]


def is_true(in_list: list):
    for temp in in_list:
        if not bool(temp):
            return False
    
    return True


'''
    key를 기준으로 정렬
        - is_reverse = False : 오름 차순
        - is_reverse = True : 내림 차순
'''
def sorted_dict_key(in_dict: dict, is_reverse=False):
    return dict(sorted(in_dict.items(), key=lambda item:item[0], reverse=is_reverse))

'''
    value를 기준으로 정렬
        - is_reverse = False : 오름 차순
        - is_reverse = True : 내림 차순
'''
def sorted_dict_value(in_dict: dict, is_reverse=False):
    return dict(sorted(in_dict.items(), key=lambda item:item[1], reverse=is_reverse))

'''
    key를 기준으로 오름 차순 정렬, value를 기준으로 내림 차순 정렬
'''
def sorted_dict(in_dict: dict):
    return sorted_dict_value(sorted_dict_key(in_dict, False), True)


def add_dict_list(in_dict: dict, key, values: list):
    if not key in in_dict.keys():
        in_dict[key] = []
    in_dict[key].extend(values)


def add_dict_set(in_dict: dict, key, values: list):
    if not key in in_dict.keys():
        in_dict[key] = set()
    in_dict[key].update(values)


def add_dict_freq(in_dict: dict, key, value=1):
    if key in in_dict.keys():
        in_dict[key] += value
    else:
        in_dict[key] = value


def get_random_key(in_dict: dict, ext_n=1):
    keys = list(in_dict.keys())
    key_len = len(keys)
    ext_n = min(ext_n, key_len)

    if ext_n == key_len:
        return keys
    else:
        result = random.sample(keys, ext_n)
        return result


def trim(input_list: list, rm_empty_flag: bool):
    if not rm_empty_flag:
        for i in range(len(input_list)):
            if input_list[i] is None:
                input_list[i] = ""
            else:
                input_list[i] = str(input_list[i]).strip()
    else:
        result = []

        for i in range(len(input_list)):
            if not is_empty(input_list[i], True):
                result.append(str(input_list[i]).strip())
        
        return result


def json_str_to_dict(json_str: str):
    try:
        # 문자열을 읽을 때는, loads() 호출
        return json.loads(json_str)

    except Exception as e:
        logging_error("json_str_to_dict()", e)
        return None


def to_json_str(input, indent=4):
    try:
        return json.dumps(input, ensure_ascii=False, indent=indent)

    except Exception as e:
        logging_error("to_json_str()", e)
        return ""


def load_json_file(in_file_path: str, encoding=ENCODING, do_print=False):
    try:
        if exists(in_file_path):
            file = open_file(in_file_path, encoding, 'r')

            # 파일을 읽을 때는, load() 호출
            datas = json.load(file)

            if do_print:
                logging(f'# falcon_util.load_json_file() datas size : {len(datas)}, in_file_path : {in_file_path}')

            return datas

    except Exception as e:
        logging_error("# falcon_util.load_json_file() error : ", e)
        return None

    return None


def write_json_file(in_json, out_file_path, encoding=ENCODING, indent=4):
    out_file = open_file(out_file_path, encoding, 'w')
    out_file.write(to_json_str(in_json, indent))
    out_file.close()


def check_gpu_memory(devices=[0], prefix='', do_print=True):
    if not torch.cuda.is_available():
        if do_print:
            logging(f'# falcon_util.check_gpu_memory() [GPU] CUDA not available')

        return -1, -1
    else:
        allocated_all, reserved_all = 0, 0

        for device in devices:
            allocated = torch.cuda.memory_allocated(device)
            reserved  = torch.cuda.memory_reserved(device)
            allocated_all += allocated
            reserved_all += reserved

            # 최대/최소 사용량 (optional)
            # max_alloc = torch.cuda.max_memory_allocated(device)
            # max_resv  = torch.cuda.max_memory_reserved(device)
            # logging(f'# falcon_util.check_gpu_memory() {prefix} [GPU:{device}] Max allocated: {max_alloc/1e9:.2f} GB | Max reserved: {max_resv/1e9:.2f} GB')

            if do_print:
                logging(f'# falcon_util.check_gpu_memory() {prefix} [GPU:{device}] Allocated: {allocated/1e9:.2f} GB | Reserved: {reserved/1e9:.2f} GB')
        if do_print:
            if len(devices) > 1:
                logging(f'# falcon_util.check_gpu_memory() {prefix} [GPU:Total] Allocated: {allocated_all/1e9:.2f} GB | Reserved: {reserved_all/1e9:.2f} GB\n')
            else:
                logging('')
        
        return allocated_all, reserved_all


def clear_gpu_memory(devices=[0], do_print=True):
    if not torch.cuda.is_available():
        if do_print:
            logging(f'# falcon_util.clear_gpu_memory() [GPU] CUDA not available')
    else:
        allocated1, reserved1 = check_gpu_memory(devices, '[Before Clear Memory]', do_print)
        gc.collect()
        torch.cuda.empty_cache()
        allocated2, reserved2 = check_gpu_memory(devices, '[After Clear Memory]', do_print)

        if do_print:
            cleared_allocated = (allocated2-allocated1)
            cleared_reserved = (reserved2-reserved1)
            logging(f'# falcon_util.clear_gpu_memory() [GPU:Total] [Cleared] Allocated: {cleared_allocated/1e9:.2f} GB | Reserved: {cleared_reserved/1e9:.2f} GB\n')

