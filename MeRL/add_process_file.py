import csv
from supar import Parser
import json


def GetTree_heads(t):
    heads = [0] * len(t)
    mapnode = [0] * len(t)

    def Findheads(cidx, t, headidx):

        if (cidx >= len(t)):
            return cidx

        mapnode[cidx] = t[cidx].lhs()
        heads[cidx] = headidx

        if t[cidx].lhs().__str__() == '_':
            mapnode[cidx] = t[cidx].rhs()[0]

            return cidx + 1

        nidx = cidx + 1
        for r in t[cidx].rhs():
            nidx = Findheads(nidx, t, cidx)

        return nidx

    Findheads(0, t, -1)
    return heads, mapnode


def get_path_and_children_dict(heads):
    path_dict = {}
    remain_nodes = list(range(len(heads)))
    delete_nodes = []

    while len(remain_nodes) > 0:
        for idx in remain_nodes:
            # 初始状态
            if idx not in path_dict:
                path_dict[idx] = [heads[idx]]  # no self
                if heads[idx] == -1:
                    delete_nodes.append(idx)  # need delete root
            else:
                last_node = path_dict[idx][-1]
                if last_node not in remain_nodes:
                    path_dict[idx].extend(path_dict[last_node])
                    delete_nodes.append(idx)
                else:
                    path_dict[idx].append(heads[last_node])
        # remove nodes
        for del_node in delete_nodes:
            remain_nodes.remove(del_node)
        delete_nodes = []

    # children_dict
    children_dict = {}
    for x, l in path_dict.items():
        if l[0] == -1:
            continue
        if l[0] not in children_dict:
            children_dict[l[0]] = [x]
        else:
            children_dict[l[0]].append(x)

    return path_dict, children_dict


def find_inner_LCA(path_dict, aspect_range):
    path_range = [[x] + path_dict[x] for x in aspect_range]
    path_range.sort(key=lambda l: len(l))

    for idx in range(len(path_range[0])):
        flag = True
        for pid in range(1, len(path_range)):
            if path_range[0][idx] not in path_range[pid]:
                flag = False
                break

        if flag:
            LCA_node = path_range[0][idx]
            break  # already find
    return LCA_node


def preprocess_file(file_path):
    print('Processing:', file_path)
    special_token = '[N]'

    dep_parser = Parser.load('biaffine-dep-en')
    con_parser = Parser.load('crf-con-en')

    sub_len = len(special_token)

    #################################################
    json_data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = csv.reader(f)
        next(lines)
        for _, line in enumerate(lines):
            sentence = {}
            sent, label, target_position, target_word, pos_tag, gloss, eg_sent = line
            sentence['sent'] = sent

            sentence['token'] = sent.split()
            token = sentence['token']
            token = [tok.replace(u'\xa0', u'') for tok in token]
            sentence['token'] = token

            # dependency parsing
            dataset = dep_parser.predict(token, verbose=False)
            dep_head = dataset.arcs[0]
            sentence['dep_head'] = [x - 1 for x in dep_head]

            # constituent parsing
            parser_inputs = ' '.join(token).replace('(', '<').replace(')', '>').split(' ')  # [ver1]

            dataset = con_parser.predict(parser_inputs, verbose=False)

            t = dataset.trees[0]

            con_head, con_mapnode = GetTree_heads(t.productions())
            sentence['con_head'] = con_head

            con_mapnode = [x if isinstance(x, str) else x.__str__() + special_token for x in con_mapnode]
            sentence['con_mapnode'] = con_mapnode

            sentence['target_position'] = target_position
            sentence['aspects'] = []
            aspect = {}
            aspect['term'] = [target_word]

            target_position = int(target_position)

            aspect['from'] = target_position
            aspect['to'] = target_position + 1
            aspect['label'] = int(label)
            sentence['aspects'].append(aspect)

            con_path_dict, con_children = get_path_and_children_dict(sentence['con_head'])

            mapback = [idx for idx, word in enumerate(con_mapnode) if word[-sub_len:] != special_token]

            for aspect_info in sentence['aspects']:
                aspect_range = list(range(mapback[aspect_info['from']], mapback[aspect_info['to'] - 1] + 1))

                con_lca = find_inner_LCA(con_path_dict, aspect_range)
                aspect_info['con_lca'] = con_lca

            sentence['pos_tag'] = pos_tag
            sentence['gloss'] = gloss
            sentence['eg_sent'] = eg_sent

            json_data.append(sentence)

    with open(file_path.replace('.csv', '.json'), 'w') as f:
        json.dump(json_data, f)

    print('done：', len(json_data))


if __name__ == '__main__':
    file_path = './data/MOH-X/MOH-X.csv'
    preprocess_file(file_path)