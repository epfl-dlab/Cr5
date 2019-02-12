import gzip
import pickle
import codecs

def get_concept_id_mapping(lang_codes):
    '''Maps the concept id to the article for that concept in English'''

    # Path to the interlanguage links file
    interlanguage_links_file = '/dlabdata1/josifosk/crosslingwiki/interlanguage_links_wikidata-20170720_WITH-INDEX.tsv.gz'
    concept_id_concept_name_mapping_per_lang = {}
    concept_name_concept_id_mapping = {}
    
    for lang_code in lang_codes:
        concept_id_concept_name_mapping_per_lang[lang_code] = {}
        
    with gzip.open(interlanguage_links_file, 'r') as f:
        c = 0
        for line in codecs.getreader("utf-8")(f):
            c += 1
            
            if c % 1000000 == 0:
                print("Processed:", c)
                
            parts = line.strip().split('\t')
            
            if parts[2] in lang_codes:
                concept_id_concept_name_mapping_per_lang[parts[2]][parts[0]] = parts[3]
                concept_name_concept_id_mapping[parts[3]] = parts[0]
                
    all_concepts = set()
    for lang_code in lang_codes:
        mappings = concept_id_concept_name_mapping_per_lang[lang_code]
        all_concepts = all_concepts.union(set(mappings.keys()))

    concept_id_concept_name_mapping = {}
    for concept_id in all_concepts:
        for lang_code in lang_codes:
            mapping = concept_id_concept_name_mapping_per_lang[lang_code]
            if concept_id in mapping:
                concept_id_concept_name_mapping[concept_id] = mapping[concept_id]
                break
                
    most_probable = concept_name_concept_id_mapping = dict((v, k) for k, v in concept_id_concept_name_mapping.items())
    for key, value in most_probable.items():
        concept_name_concept_id_mapping[key] = value
        
    return concept_id_concept_name_mapping, concept_name_concept_id_mapping

lang_codes = ['simple', 'en', 'de', 'es', 'it', 'eu', 'ru', 'fr' 'mk', 'nl', 'el', 'no', 'sl', 'bg', 'pl', 'ca', 'et', 'hu', 'pt', 'sv', 'hr', 'fi', 'id', 'ro', 'tr', 'cs', 'uk', 'da', 'sk', 'vi', 'da']
concept_id_concept_name_mapping, concept_name_concept_id_mapping = get_concept_id_mapping(lang_codes)


pickle.dump(concept_id_concept_name_mapping, open('concept_id_concept_name_mapping.p', 'wb'))
pickle.dump(concept_name_concept_id_mapping, open('concept_name_concept_id_mapping.p', 'wb'))