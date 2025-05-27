from .falcon_util import *

# !pip install SPARQLWrapper
from SPARQLWrapper import SPARQLWrapper, JSON


def get_sparql():
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.addCustomHttpHeader("User-Agent", "Mozilla/5.0")

    return sparql


'''
    주어진 subject_label과 relation(PID)에 대응되는 object label을 반환
        - subject_label: 위키데이터 라벨 (예: 'Eibenstock')
        - relation_pid: Wikidata property (예: 'P17')
        - lang: 라벨 언어 (기본: 영어)
'''
def get_object_of_subject_and_relation(subject_label: str, relation_pid: str, limit=1, lang: str='en', sparql=None):
    if sparql is None:
        sparql = get_sparql()

    query = f"""
        SELECT DISTINCT ?object ?objectLabel WHERE {{
            ?subject rdfs:label "{subject_label}"@{lang} .
            ?subject wdt:{relation_pid} ?object .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang}" . }}
    }}
    LIMIT {limit}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    # Execute the query
    try:
        results = sparql.query().convert()
    except Exception as e:
        print(f'\n\n### (error) wiki_util.get_object_of_subject_and_relation() error msg : {e}\n\n')
        return None, None
    
    object_labels = [res["objectLabel"]["value"] for res in results["results"]["bindings"]]
    object_pids = [res["object"]["value"] for res in results["results"]["bindings"]]

    return object_labels, object_pids


def _test_single(subject_labels, relation_pids):
    sparql = get_sparql()

    for subject_label in subject_labels:
        for relation_pid in relation_pids:
            print(f'subject_label : {subject_label}, relation_pid : {relation_pid}')
            object_labels, object_pids = get_object_of_subject_and_relation(subject_label, relation_pid, 10, sparql=sparql)

            for i, object_label in enumerate(object_labels):
                print(f'[{i}] object_label : {object_label}, object_pid : {object_pids[i]}')
            print()


if __name__ == "__main__" :
    subject_labels = ['Eiffel Tower', 'Falash Mura']
    relation_pids = ['P127', 'P276']

    _test_single(subject_labels, relation_pids)

