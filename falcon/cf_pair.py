from typing import Any, Dict, List, Set, Union, Optional


def get_data_template(null_int=-1, null_str='') -> dict:
    data = {
        'case_id': null_int,
        'pararel_idx': null_int,
        'requested_rewrite': {
            'prompt': null_str,
            'relation_id': null_str,
            'target_new': {
                'str': null_str,
                'id': null_str
            },
            'target_true': {
                'str': null_str,
                'id': null_str
            },
            'subject': null_str
        },
        'paraphrase_prompts': [],
        'neighborhood_prompts': [],
        'attribute_prompts': [],
        'generation_prompts': []
    }

    return data


class CFPair:
    def __init__(self, data: dict):
        self._case_id: Optional[int] = None
        self._pararel_idx: Optional[int] = None
        
        self._prompt: str = None
        self._relation_id: str = None
        
        self._target_new_str: str = None
        self._target_new_id: str = None
        self._target_true_str: str = None
        self._target_true_id: str = None

        self._subject: str = None

        self._paraphrase_prompts: List[str] = None
        self._neighborhood_prompts: List[str] = None
        self._attribute_prompts: List[str] = None
        self._generation_prompts: List[str] = None

        self._parsing(data)
    

    def _parsing(self, data: Dict[str, Any]):
        def _assign(d: Dict[str, Any], prefix: str = ""):
            for key, value in d.items():
                # 1) 최상위 requested_rewrite 는 그룹핑만 할 뿐 prefix에 포함시키지 않는다
                if isinstance(value, dict):
                    if key == "requested_rewrite":
                        _assign(value, prefix)
                    else:
                        # 2) 'target_new', 'target_true' 등부터는 prefix=key
                        _assign(value, key)
                else:
                    # 3) str/id 등 실제 값 할당 시
                    if prefix:
                        attr = f"_{prefix}_{key}"
                    else:
                        attr = f"_{key}"
                    if hasattr(self, attr):
                        setattr(self, attr, value)

        _assign(data)


    def validate(self):
        missing = [name for name, val in vars(self).items() if val is None]

        if missing:
            print(f'### (ERROR) CFPair.validate() The following member variables are not set : {", ".join(missing)}')
            return False

        return True


    def _print(self):
        if self.validate():
            print(f'\n# CFPair._print()')
            print(f'case_id : {self._case_id}')
            print(f'pararel_idx : {self._pararel_idx}')
            print(f'prompt : {self._prompt}')
            print(f'relation_id : {self._relation_id}')
            print(f'target_new : {self._target_new_str} ( {self._target_new_id} )')
            print(f'target_true : {self._target_true_str} ( {self._target_true_id} )')
            print(f'subject : {self._subject}')
            print(f'paraphrase_prompts : {self._paraphrase_prompts}')
            print(f'neighborhood_prompts : {self._neighborhood_prompts}')
            print(f'attribute_prompts : {self._attribute_prompts}')
            print(f'generation_prompts : {self._generation_prompts}\n')

