import string
from collections import Counter
from typing import Dict, List, Tuple, Union, Callable

import nltk
import numpy as np
import math
import pandas as pd
import torch
import torch.nn.functional as F



glue_qqp_dir = '.'
glove_path = './glove.6B.50d.txt'



class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(-1 * (x - self.mu) ** 2 / (2 * self.sigma ** 2))



class KNRM(torch.nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, freeze_embeddings: bool, kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()

        self.out_activation = torch.nn.Sigmoid()


    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        """инициализируем ядра с соответствующими ню и сигма"""
        kernels = torch.nn.ModuleList()
        # создаем список mu
        distance = 2 / (self.kernel_num - 1) # distanse between kernels
        first_kernel = - distance / 2 - distance * ((self.kernel_num - 1) / 2 - 1)
        mu_list = [first_kernel + distance * i for i in range(self.kernel_num - 1)]
        mu_list.append(1)

        # создаем ядра
        for cur_kernel_idx in range(self.kernel_num - 1):
            cur_kernel = GaussianKernel(mu=mu_list[cur_kernel_idx], sigma=self.sigma)
            kernels.append(cur_kernel)
        last_kernel = GaussianKernel(mu=1, sigma=self.exact_sigma)
        kernels.append(last_kernel)
        return kernels


    def _get_mlp(self) -> torch.nn.Sequential:
        """создаем MLP"""
        layers = []
        cur_size = self.kernel_num # размерность начального вектора == количество ядер (K)
        for new_size in self.out_layers:
            layers.append(torch.nn.Linear(cur_size, new_size))
            cur_size = new_size
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(cur_size, 1))
        return torch.nn.Sequential(*layers)


    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out


    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        return F.cosine_similarity(query.unsqueeze(1), doc.unsqueeze(0), dim=2)


    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out


    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        # shape = [Batch, Left], [Batch, Right]
        query, doc = inputs['query'], inputs['document']

        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out



class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, index_pairs_or_triplets: List[List[Union[str, float]]],
                 idx_to_text_mapping: Dict[str, str], vocab: Dict[str, int], oov_val: int,
                 preproc_func: Callable, max_len: int = 30):
        self.index_pairs_or_triplets = index_pairs_or_triplets      # список троек : [[query_id, document_id, label]...] для валидации
                                                                    # или четверок [[query_id, document_1_id, document_2_id, label]...] для трейна
        self.idx_to_text_mapping = idx_to_text_mapping              # словарь - id вопроса: вопрос
        self.vocab = vocab                                          # словарь - токен: №строки в матрице
        self.oov_val = oov_val                                      # индекс в словаре эмбеддингов для слова которого не было в трэйне
        self.preproc_func = preproc_func                            # предобработка и токенизация слов
        self.max_len = max_len                                      # максимальное число токенов в тексте

    def __len__(self):
        return len(self.index_pairs_or_triplets)

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        """список токенов -> список id токенов в словаре с №строк в матрице эмбеддингов"""
        return [self.vocab[token] for token in tokenized_text]

    def _convert_text_idx_to_token_idxs(self, idx: int) -> List[int]:
        """id вопроса (id_left или id_right) -> ids токенов"""
        text = self.idx_to_text_mapping[idx]
        tokenized_text = self.preproc_func(text)
        return self._tokenized_text_to_index(tokenized_text)

    def __getitem__(self, idx: int):
        pass


class TrainTripletsDataset(RankingDataset):
    def __getitem__(self, idx):
        """label — ответ на вопрос, действительно ли первый документ более релевантен запросу, чем второй"""
        query_id, document_1_id, document_2_id, label = self.index_pairs_or_triplets
        query = self._convert_text_idx_to_token_idxs(query_id)
        document_1 = self._convert_text_idx_to_token_idxs(document_1_id)
        document_2 = self._convert_text_idx_to_token_idxs(document_2_id)
        return {'query': query, 'document': document_1}, {'query': query, 'document': document_2}, label


class ValPairsDataset(RankingDataset):
    def __getitem__(self, idx):
        """label - релевантность от 0 до 2"""
        query_id, document_id, label = self.index_pairs_or_triplets[idx]
        query = self._convert_text_idx_to_token_idxs(query_id)
        document = self._convert_text_idx_to_token_idxs(document_id)
        return {'query': query, 'document': document}, label



def collate_fn(batch_objs: List[Union[Dict[str, torch.Tensor], torch.FloatTensor]]):
    """обработка пакета данных, где каждый элемент пакета может быть либо парой, либо триплетом. 
       Функция дополняет последовательности, чтобы обеспечить, 
       что все элементы в пакете имеют одинаковую длину"""
    max_len_q1 = -1
    max_len_d1 = -1
    max_len_q2 = -1
    max_len_d2 = -1

    is_triplets = False
    for elem in batch_objs:
        if len(elem) == 3:
            left_elem, right_elem, label = elem
            is_triplets = True
        else:
            left_elem, label = elem

        max_len_q1 = max(len(left_elem['query']), max_len_q1)
        max_len_d1 = max(len(left_elem['document']), max_len_d1)
        if len(elem) == 3:
            max_len_q2 = max(len(right_elem['query']), max_len_q2)
            max_len_d2 = max(len(right_elem['document']), max_len_d2)

    q1s = []
    d1s = []
    q2s = []
    d2s = []
    labels = []

    for elem in batch_objs:
        if is_triplets:
            left_elem, right_elem, label = elem
        else:
            left_elem, label = elem

        pad_len1 = max_len_q1 - len(left_elem['query'])
        pad_len2 = max_len_d1 - len(left_elem['document'])
        if is_triplets:
            pad_len3 = max_len_q2 - len(right_elem['query'])
            pad_len4 = max_len_d2 - len(right_elem['document'])

        q1s.append(left_elem['query'] + [0] * pad_len1)
        d1s.append(left_elem['document'] + [0] * pad_len2)
        if is_triplets:
            q2s.append(right_elem['query'] + [0] * pad_len3)
            d2s.append(right_elem['document'] + [0] * pad_len4)
        labels.append([label])
    q1s = torch.LongTensor(q1s)
    d1s = torch.LongTensor(d1s)
    if is_triplets:
        q2s = torch.LongTensor(q2s)
        d2s = torch.LongTensor(d2s)
    labels = torch.FloatTensor(labels)

    ret_left = {'query': q1s, 'document': d1s}
    if is_triplets:
        ret_right = {'query': q2s, 'document': d2s}
        return ret_left, ret_right, labels
    else:
        return ret_left, labels




class Solution:
    def __init__(self, glue_qqp_dir: str, glove_vectors_path: str,
                 min_token_occurancies: int = 1,
                 random_seed: int = 0,
                 emb_rand_uni_bound: float = 0.2,
                 freeze_knrm_embeddings: bool = True,
                 knrm_kernel_num: int = 21,
                 knrm_out_mlp: List[int] = [],
                 dataloader_bs: int = 1024,
                 train_lr: float = 0.001,
                 change_train_loader_ep: int = 10
                 ):
        self.glue_qqp_dir = glue_qqp_dir
        self.glove_vectors_path = glove_vectors_path
        self.glue_train_df = self.get_glue_df('train')
        self.glue_dev_df = self.get_glue_df('dev')
        self.dev_pairs_for_ndcg = self.create_val_pairs(self.glue_dev_df)
        self.min_token_occurancies = min_token_occurancies
        self.all_tokens = self.get_all_tokens(
            [self.glue_train_df, self.glue_dev_df], self.min_token_occurancies)

        self.random_seed = random_seed
        self.emb_rand_uni_bound = emb_rand_uni_bound
        self.freeze_knrm_embeddings = freeze_knrm_embeddings
        self.knrm_kernel_num = knrm_kernel_num
        self.knrm_out_mlp = knrm_out_mlp
        self.dataloader_bs = dataloader_bs
        self.train_lr = train_lr
        self.change_train_loader_ep = change_train_loader_ep

        self.model, self.vocab, self.unk_words = self.build_knrm_model()
        self.idx_to_text_mapping_train = self.get_idx_to_text_mapping(
            self.glue_train_df)
        self.idx_to_text_mapping_dev = self.get_idx_to_text_mapping(
            self.glue_dev_df)

        self.val_dataset = ValPairsDataset(self.dev_pairs_for_ndcg,
              self.idx_to_text_mapping_dev,
              vocab=self.vocab, oov_val=self.vocab['OOV'],
              preproc_func=self.simple_preproc)
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.dataloader_bs, num_workers=0,
            collate_fn=collate_fn, shuffle=False)


    def get_glue_df(self, partition_type: str) -> pd.DataFrame:
        assert partition_type in ['dev', 'train']
        glue_df = pd.read_csv(
            self.glue_qqp_dir + f'/{partition_type}.tsv', sep='\t', dtype=object)
        glue_df = glue_df.dropna(axis=0, how='any').reset_index(drop=True) # удаляем строки с хотя бы одним пропущенным значением
        glue_df_fin = pd.DataFrame({
            'id_left': glue_df['qid1'],
            'id_right': glue_df['qid2'],
            'text_left': glue_df['question1'],
            'text_right': glue_df['question2'],
            'label': glue_df['is_duplicate'].astype(int)
        })
        return glue_df_fin


    def handle_punctuation(self, inp_str: str) -> str:
        """удаляем знаки пунктуации"""
        translation_table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        cleaned_string = inp_str.translate(translation_table)
        return cleaned_string


    def simple_preproc(self, inp_str: str) -> List[str]:
        """предобработка и токенизация слов"""
        string_without_punctuation = self.handle_punctuation(inp_str)
        return nltk.word_tokenize(string_without_punctuation.lower())


    def _filter_rare_words(self, vocab: Dict[str, int], min_occurancies: int) -> Dict[str, int]:
        return {word: vocab[word] for word in vocab if vocab[word] >= min_occurancies}


    def get_all_tokens(self, list_of_df: List[pd.DataFrame], min_occurancies: int) -> List[str]:
        """получаем все токены датафрейма"""
        token_frequency_dict = {}
        for df in list_of_df:
            for text in set(df['text_left']) | set(df['text_right']):
                tokens = self.simple_preproc(text) # type(tokens) = list
                for token in tokens:
                    token_frequency_dict[token] = token_frequency_dict.get(token, 0) + 1
        acceptable_tokens = list(self._filter_rare_words(token_frequency_dict, min_occurancies))
        return acceptable_tokens


    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        """словарь - слово: эмбеддинг"""
        with open(file_path, 'r') as file:
            word_embedding_dict = {}
            for line in file:
                parts = line.split()
                word = parts[0]
                embedding = list(map(float, parts[1:]))
                word_embedding_dict[word] = embedding
        return word_embedding_dict


    def create_glove_emb_from_file(self, file_path: str, inner_keys: List[str],
                                   random_seed: int, rand_uni_bound: float
                                   ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
        """vocab : словарь с ключами - токенами, значениями - индекс строки в матрице эмбэддингов
                   для всех токенов на которых тренировалась модель (len = (N+2))
           emb   : матрица эбэддингов, по строкам - эмбэддинги слов
           unk_words : токены которых не было в скачанном датасете эмбеддингов GloVe
        """
        np.random.seed(random_seed)
        token_embedding_dict = self._read_glove_embeddings(file_path) # известные токены
        # inner_keys == self.all_tokens
        unk_words = ['PAD', 'OOV'] + [token for token in inner_keys if token not in token_embedding_dict]
        vocab = {'PAD': 0, 'OOV': 1}
        vocab.update({token: index for index, token in enumerate(inner_keys, start=2)}) # len = N + 2
        emb_matrix_1 = np.array(
            [np.zeros(50), np.random.uniform(low=-rand_uni_bound, high=rand_uni_bound, size=50)]
        )
        emb_matrix_main = np.array([
            token_embedding_dict[cur_token] if cur_token in token_embedding_dict 
            else np.random.uniform(low=-rand_uni_bound, high=rand_uni_bound, size=50)
            for cur_token in inner_keys
            ]) # матрица: для известных слов - эмбеддинги, для неизвестных - случайные вектора

        emb_matrix = np.vstack((emb_matrix_1, emb_matrix_main)) # dim == (N + 2, 50)
        return emb_matrix, vocab, unk_words


    def build_knrm_model(self) -> Tuple[torch.nn.Module, Dict[str, int], List[str]]:
        emb_matrix, vocab, unk_words = self.create_glove_emb_from_file(
            self.glove_vectors_path, self.all_tokens, self.random_seed, self.emb_rand_uni_bound)
        torch.manual_seed(self.random_seed)
        knrm = KNRM(emb_matrix, freeze_embeddings=self.freeze_knrm_embeddings,
                    out_layers=self.knrm_out_mlp, kernel_num=self.knrm_kernel_num)
        return knrm, vocab, unk_words


    def sample_data_for_train_iter(self, inp_df: pd.DataFrame, seed: int, one_query_quantity: int = 4,
                                    min_group_size: int = 4, sample_length: int = 8000) -> List[List[Union[str, float]]]:
        """на выходе список : [[query_id, document_1_id, document_2_id, label]...]
            label — ответ на вопрос, действительно ли первый документ более релевантен запросу, чем второй
            Хотим нагенерировать ~10000 примеров для тренировки.
            label = 0: релевантности (0, 1), (0, 2), (1, 2) таких ~5000 примеров
            label = 1: релевантности (1, 0), (2, 0), (2, 1) таких ~5000 примеров
        """
        # one_query_quantity = 4  # сколько в выборке будет примеров с одним query_id
        # min_group_size = 4
        # sample_length = 10000

        inp_df_select = inp_df[['id_left', 'id_right', 'label']]
        inf_df_group_sizes = inp_df_select.groupby('id_left').size()
        glue_dev_leftids_to_use = list(
            inf_df_group_sizes[inf_df_group_sizes >= min_group_size].index)
        groups = inp_df_select[inp_df_select.id_left.isin(
            glue_dev_leftids_to_use)].groupby('id_left')

        all_ids = set(inp_df['id_left']).union(set(inp_df['id_right']))

        out_pairs = []

        np.random.seed(seed)
        
        num_pairs_one_relevant = 0 # число пар документов с одним релевантным (01 или 02)
        for query_id, group in groups:
            """в group все релевантные документы для запроса query_id
            для каждого query_id выберем по one_query_quantity / 2 триплетов с label = 1 и label = 0"""
            ones_ids = group[group.label > 0].id_right.values
            zeroes_ids = group[group.label == 0].id_right.values
            cur_chosen = set(ones_ids).union(set(zeroes_ids)).union({id_left})

            if len(out_pairs) > sample_length:
                break

            if (
                len(ones_ids) * len(zeroes_ids) == 0 and 
                num_pairs_one_relevant < sample_length * 0.6
                ):
                num_pairs_one_relevant += 1
                # если в группе все либо полные дупликаты либо все просто очень похожи
                # в этом случае в каждом триплете будет нерелевантный документ
                not_relevant_docs = np.random.choice(
                    list(all_ids - cur_chosen), one_query_quantity, replace=False).tolist()
                relevant_docs = np.random.choice(
                    list(cur_chosen), one_query_quantity, replace=True).tolist()
                for i in range(one_query_quantity//2):
                    out_pairs.append([query_id, relevant_docs[i], not_relevant_docs[i], 1])
                for i in range(one_query_quantity//2, one_query_quantity):
                    out_pairs.append([query_id, not_relevant_docs[i], relevant_docs[i], 0])

            elif len(ones_ids) * len(zeroes_ids) == 0:
                # в группе есть как полные дупликаты так и просто очень похожие
                local_counter = 0
                for one_rel_doc in ones_ids:
                    for zero_rel_doc in zeroes_ids:
                        if local_counter % 2 == 0:
                            out_pairs.append([query_id, one_rel_doc, zero_rel_doc, 1])
                        else:
                            out_pairs.append([query_id, zero_rel_doc, one_rel_doc, 0])
        return out_pairs


    def create_val_pairs(self, inp_df: pd.DataFrame, fill_top_to: int = 15,
                         min_group_size: int = 2, seed: int = 0) -> List[List[Union[str, float]]]:
        """на выходе список троек : [[id_left, id_right, label]...]
           fill_top_to - какое минимальное число раз id_left должно встретиться в валидационной выборке (в качестве запроса)"""
        inp_df_select = inp_df[['id_left', 'id_right', 'label']]
        inf_df_group_sizes = inp_df_select.groupby('id_left').size()
        glue_dev_leftids_to_use = list(
            inf_df_group_sizes[inf_df_group_sizes >= min_group_size].index)
        groups = inp_df_select[inp_df_select.id_left.isin(
            glue_dev_leftids_to_use)].groupby('id_left')

        all_ids = set(inp_df['id_left']).union(set(inp_df['id_right']))

        out_pairs = []

        np.random.seed(seed)

        for id_left, group in groups:
            """в group все релевантные документы для запроса id_left"""
            ones_ids = group[group.label > 0].id_right.values
            zeroes_ids = group[group.label == 0].id_right.values
            sum_len = len(ones_ids) + len(zeroes_ids)
            num_pad_items = max(0, fill_top_to - sum_len)
            if num_pad_items > 0:
                cur_chosen = set(ones_ids).union(
                    set(zeroes_ids)).union({id_left})
                pad_sample = np.random.choice(
                    list(all_ids - cur_chosen), num_pad_items, replace=False).tolist()
            else:
                pad_sample = []
            for i in ones_ids:
                out_pairs.append([id_left, i, 2])
            for i in zeroes_ids:
                out_pairs.append([id_left, i, 1])
            for i in pad_sample:
                out_pairs.append([id_left, i, 0])
        return out_pairs


    def get_idx_to_text_mapping(self, inp_df: pd.DataFrame) -> Dict[str, str]:
        """на выходе словарь - id вопроса: вопрос"""
        left_dict = (
            inp_df
            [['id_left', 'text_left']]
            .drop_duplicates()
            .set_index('id_left')
            ['text_left']
            .to_dict()
        )
        right_dict = (
            inp_df
            [['id_right', 'text_right']]
            .drop_duplicates()
            .set_index('id_right')
            ['text_right']
            .to_dict()
        )
        left_dict.update(right_dict)
        return left_dict


    def ndcg_k(self, ys_true: np.array, ys_pred: np.array, ndcg_top_k: int = 10) -> float:
        # (внимание, используются вектора numpy)
        pass


    def valid(self, model: torch.nn.Module, val_dataloader: torch.utils.data.DataLoader) -> float:
        labels_and_groups = val_dataloader.dataset.index_pairs_or_triplets
        labels_and_groups = pd.DataFrame(labels_and_groups, columns=['left_id', 'right_id', 'rel'])

        all_preds = []
        for batch in (val_dataloader):
            inp_1, y = batch
            preds = model.predict(inp_1)
            preds_np = preds.detach().numpy()
            all_preds.append(preds_np)
        all_preds = np.concatenate(all_preds, axis=0)
        labels_and_groups['preds'] = all_preds

        ndcgs = []
        for cur_id in labels_and_groups.left_id.unique():
            cur_df = labels_and_groups[labels_and_groups.left_id == cur_id]
            ndcg = self.ndcg_k(cur_df.rel.values.reshape(-1), cur_df.preds.values.reshape(-1))
            if np.isnan(ndcg):
                ndcgs.append(0)
            else:
                ndcgs.append(ndcg)
        return np.mean(ndcgs)

    def train(self, n_epochs: int):
        opt = torch.optim.SGD(self.model.parameters(), lr=self.train_lr)
        criterion = torch.nn.BCELoss()
        pass



def main():
    solution = Solution(glue_qqp_dir=glue_qqp_dir, glove_vectors_path=glove_path)
    print(solution.test1)
    print()
    print(solution.test2)
    print()
    print(solution.test3)
    


if __name__ == "__main__":
    main()