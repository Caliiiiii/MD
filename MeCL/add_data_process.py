import csv
from transformers import AutoTokenizer
import os
import json
from add_spans import *
import numpy as np
from copy import copy, deepcopy


class Data_Processor:
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.DATA.plm)
        self.data_dir = args.DATA.data_dir
        self.sep_puncs = args.DATA.sep_puncs
        self.max_left_len = args.DATA.max_left_len  # the previously set max length of the left input
        self.max_right_len = args.DATA.max_right_len
        self.use_pos = args.DATA.use_pos
        self.use_eg_sent = args.DATA.use_eg_sent
        self.use_def = args.DATA.use_def  
        self.max_hop = args.DATA.max_hop  

    def __str__(self):
        pattern = '''Data Configs: 
        data_dir: {} 
        sep_puncs: {} 
        max_left_len: {} 
        max_right_len: {} 
        use_pos: {}
        use_eg_sent: {}'''
        return pattern.format(self.data_dir, self.sep_puncs, self.max_left_len,
                              self.max_right_len, self.use_pos, self.use_eg_sent,
                              self.use_def, self.max_hop)

    ######################################################################

    def _get_examples(self, file_dir):
        # load_data
        with open(file_dir, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed = []

        sep_puncs = [self.tokenizer.encode(sep_punc, add_special_tokens=False)[0] for sep_punc in self.sep_puncs]

        for d in data:
            # segment ids: 1==>target word, 2==>local context, 3==>pos, 0==>others
            label = d['label']  
            target_position = d['target_position']  
            sep_id = self.tokenizer.encode(self.tokenizer.sep_token, add_special_tokens=False)[0]
            """
            for the left part
            """
            # convert sentence to ids: <s> sentence </s> 
            ids_l = self.tokenizer.encode(d['sent'])
            segs_l = [0] * len(ids_l)

            # target word may be cut into word pieces by tokenizer, find the range of pieces
            tar_start, tar_end = target_align(target_position, d['sent'], tokenizer=self.tokenizer)
            local_start, local_end = get_local(ids_l, tar_start, sep_puncs)

            # set local segments and target segments
            segs_l[local_start: local_end] = [2] * (local_end - local_start)
            segs_l[tar_start: tar_end] = [1] * (tar_end - tar_start)

            # segment ids: 1==>target word, 2==>local context, 3==>pos, 0==>others
            '''
            for the right part:<s> target_words </s> </s> POS </s> </s> definition </s>
            '''
            ids_r = self.tokenizer.encode(d['target_word'])  # <s> target_word </s>
            segs_r = [0] * len(ids_r)
            segs_r[1:-1] = [1] * (len(ids_r) - 2)  # except <s> and </s>, rest tokens are tagged as target_word

            # -- use_pos：whether to use pos tag 
            if self.use_pos:
                # </s> POS </S>
                pos_ids = [sep_id] + self.tokenizer.encode(d['pos_tag'], add_special_tokens=False) + [sep_id]
                pos_segs = [0] + [3] * (len(pos_ids) - 2) + [0]  
                ids_r = ids_r + pos_ids
                segs_r = segs_r + pos_segs

            if self.use_def and d['gloss'] != '':
                # </s> basic_usage </s>
                def_ids = [sep_id] + self.tokenizer.encode(d['gloss'], add_special_tokens=False) + [sep_id]
                def_segs = [0] + [2] * (len(def_ids) - 2) + [0]
                ids_r = ids_r + def_ids
                segs_r = segs_r + def_segs

            left_len = len(ids_l)
            right_len = len(ids_r)

            assert left_len == len(segs_l)
            assert right_len == len(segs_r)

            pad_id = self.tokenizer.encode(self.tokenizer.pad_token, add_special_tokens=False)[0]

            if left_len < self.max_left_len:  # left_len = 12 , max_left_len = 25
                res = self.max_left_len - left_len  # residual
                ids_l = ids_l + [pad_id] * res  
                segs_l = segs_l + [0] * res
                att_mask_l = [1] * left_len + [0] * res
            else:
                ids_l = ids_l[:self.max_left_len]
                segs_l = segs_l[:self.max_left_len]
                att_mask_l = [1] * self.max_left_len

            if right_len < self.max_right_len:  # right_len = 16,max_right_len = 70
                res = self.max_right_len - right_len  # 54
                ids_r = ids_r + [pad_id] * res
                segs_r = segs_r + [0] * res
                att_mask_r = [1] * right_len + [0] * res
            else:
                ids_r = ids_r[:self.max_right_len]
                segs_r = segs_r[:self.max_right_len]
                att_mask_r = [1] * self.max_right_len

            frm = target_position
            to = target_position + 1
            dep_dir = self.reshape_new(frm, to, d['dependencies'],
                                       self.max_hop, d['tokens'])

            neighbor_mask = []
            for idx, word in enumerate(d['sent'].lower().split()):
                word_tokens = self.tokenizer.tokenize(word)
                if dep_dir[idx] != 0:
                    neighbor_mask.extend(np.ones(len(word_tokens)))
                else:
                    neighbor_mask.extend(np.zeros(len(word_tokens)))  # -- [0.0, 0.0, 0.0,

            nb_len = len(neighbor_mask)
            if nb_len < self.max_left_len:
                res = self.max_left_len - nb_len
                neighbor_mask = neighbor_mask + [0] * res
            else:
                neighbor_mask = neighbor_mask[:self.max_left_len]

            example = [ids_l, segs_l, att_mask_l, ids_r, att_mask_r, segs_r, neighbor_mask, label]
            processed.append(example)

        return processed

    def reshape_new(self, as_start, as_end, dependencies, max_hop, tokens=None):
        dep_idx = []
        dep_dir = []
        # 1 hop
        for i in range(as_start, as_end):  
            for dep in dependencies:  
                if i == dep[1] - 1:
                    if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                        if str(dep[0]) != 'punct':  
                            dep_dir.append(1)  
                        else:
                            dep_dir.append(0)
                        dep_idx.append(dep[2] - 1)  
                elif i == dep[2] - 1:
                    if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                        if str(dep[0]) != 'punct':  
                            dep_dir.append(2)  # -- 2
                        else:
                            dep_dir.append(0)
                        dep_idx.append(dep[1] - 1)  

        current_hop = 2
        added = True
        while current_hop <= max_hop and len(dep_idx) < len(tokens) and added:  
            added = False  # -- ！！
            dep_idx_temp = deepcopy(dep_idx)  
            for i in dep_idx_temp:
                for dep in dependencies:  
                    if i == dep[1] - 1:  
                        if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[
                            2] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  
                                dep_dir.append(1)  
                            else:
                                dep_dir.append(0)
                            dep_idx.append(dep[2] - 1)  # -- [46, 52, 45]
                            added = True
                    elif i == dep[2] - 1:
                        # not root, not aspect
                        if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[
                            1] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                                dep_dir.append(2)  # -- [2, 1, 1, 2]
                            else:
                                dep_dir.append(0)
                            dep_idx.append(dep[1] - 1)  # -- [46, 52, 45, 44]
                            added = True
            current_hop += 1  

        for idx, token in enumerate(tokens):  
            if idx not in dep_idx and (idx < as_start or idx >= as_end):  
                dep_dir.append(0)
                dep_idx.append(idx)

        for idx, token in enumerate(tokens):  
            if idx not in dep_idx:
                dep_dir.append(0)
                dep_idx.append(idx)

        index = [i[0] for i in
                 sorted(enumerate(dep_idx), key=lambda x: x[1])]  
        dep_dir = [dep_dir[i] for i in index]

        assert len(tokens) == len(dep_idx), 'length wrong'
        return dep_dir


class VUA_All_Processor(Data_Processor):
    def __init__(self, args):
        super(VUA_All_Processor, self).__init__(args)

    def get_train_data(self):
        train_data_path = os.path.join(self.data_dir, 'VUA_All/train.json')
        train_data = self._get_examples(train_data_path)
        return train_data

    def get_val_data(self):
        val_data_path = os.path.join(self.data_dir, 'VUA_All/val.json')
        val_data = self._get_examples(val_data_path)
        return val_data

    def get_test_data(self):
        test_data_path = os.path.join(self.data_dir, 'VUA_All/test.json')
        test_data = self._get_examples(test_data_path)
        return test_data

    def get_acad(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/genre/acad.json')
        data = self._get_examples(data_path)
        return data

    def get_conv(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/genre/conv.json')
        data = self._get_examples(data_path)
        return data

    def get_fict(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/genre/fict.json')
        data = self._get_examples(data_path)
        return data

    def get_news(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/genre/news.json')
        data = self._get_examples(data_path)
        return data

    def get_adj(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/pos/adj.json')
        data = self._get_examples(data_path)
        return data

    def get_adv(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/pos/adv.json')
        data = self._get_examples(data_path)
        return data

    def get_noun(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/pos/noun.json')
        data = self._get_examples(data_path)
        return data

    def get_verb(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/pos/verb.json')
        data = self._get_examples(data_path)
        return data


class Verb_Processor(Data_Processor):

    def __init__(self, args):
        super(Verb_Processor, self).__init__(args)

    def get_train_data(self):
        data_path = os.path.join(self.data_dir, 'VUA_Verb/new_train.json')
        data = self._get_examples(data_path)
        return data

    def get_val_data(self):
        data_path = os.path.join(self.data_dir, 'VUA_Verb/new_val.json')
        data = self._get_examples(data_path)
        return data

    def get_test_data(self):
        data_path = os.path.join(self.data_dir, 'VUA_Verb/new_test.json')
        data = self._get_examples(data_path)
        return data

    def get_trofi(self):
        data_path = os.path.join(self.data_dir, 'TroFi/new_TroFi.json')
        data = self._get_examples(data_path)
        return data

    def get_mohx(self):
        data_path = os.path.join(self.data_dir, 'MOH-X/new_MOH-X.json')
        data = self._get_examples(data_path)
        return data


def get_local(tokens, target_start, sep_puncs):
    """
    A local context is the clause that the target word occurs. Use sep_puncs to split different clauses.

    :param tokens: (list) a tokenized sentence
    :param target_start: (int) the start idx of the target_word
    :param sep_puncs: (list) all the punctuations that split a context
    :return: (tuple of int) the start idx and end idx of local context
    """
    local_start = 1
    local_end = local_start + len(tokens)
    for i, w in enumerate(tokens):
        if i < target_start and w in sep_puncs:
            local_start = i + 1
        if i > target_start and w in sep_puncs:
            local_end = i
            break
    return local_start, local_end


def target_align(target_position, sentence, tokenizer):
    """
    A target may be cut into word pieces by Tokenizer, this func tries to find the start and end idx of the target word
    after tokenization.
    NOTICE: we return a half-closed range. eg. [0, 6) for start_idx=0 and end_idx=6

    :param target_position: (int) the position of the target word in the original sentence
    :param sentence: (string) original sentence
    :param tokenizer: an instance of Transformers Tokenizer
    :return: (tuple of int) the start and end idx of the target word in the tokenized form
    """
    start_idx = 1  
    end_idx = 0
    for j, word in enumerate(sentence.split()):
        if not j == 0:  
            word = ' ' + word
        word_tokens = tokenizer.tokenize(word)  
        if not j == target_position:  
            start_idx += len(word_tokens)
        else:  # else, calculate the end position
            end_idx = start_idx + len(word_tokens)
            break  # once find, stop looping.
    return start_idx, end_idx


def text2bert_id(token, tokenizer): 
    re_token = []  
    word_mapback = []  
    word_split_len = [] 
    for idx, word in enumerate(token):
        temp = tokenizer.tokenize(word)
        re_token.extend(temp)
        word_mapback.extend([idx] * len(temp))
        word_split_len.append(len(temp))
    re_id = tokenizer.convert_tokens_to_ids(re_token)
    return re_id, word_mapback, word_split_len
