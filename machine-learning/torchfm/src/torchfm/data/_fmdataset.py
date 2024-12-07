import bisect
import torch
from loguru import logger
from torch import Tensor
from torch.utils.data import Dataset
from torchfm.data.data_utils import encode

class FMDataset(Dataset):
    def __init__(
        self, 
        interactions: list[tuple[int, int, float]],
        user_info: list[tuple[int, list[str]]],
        item_info: list[tuple[int, list[str]]],
        user_features: dict[str, int],
        item_features: dict[str, int],
        use_cache: bool=True,
        verbose: bool=False
    ) -> None:
        self.user_input_dim = len(user_features)
        self.item_input_dim = len(item_features)
        self._interactions = interactions
        self._user_info = sorted(user_info, key=lambda x: x[0])
        self._item_info = sorted(item_info, key=lambda x: x[0])
        self._user_features = user_features
        self._item_features = item_features
        self._verbose = verbose
        self._cache_enabled = use_cache
        if use_cache:
            self._cache = {}

    def transform(
        self, 
        user_features: list[str],
        item_features: list[str],
        user_id: int = None, item_id: int = None
    ):
        if user_id is not None:
            user_features = list(set(user_features + [user_id]))

        if item_id is not None:
            item_features = list(set(item_features + [item_id]))
            
        u = encode(user_features, self._user_features)
        i = encode(item_features, self._item_features)
        return u, i
    
    def get_interacted_items(self, user_id) -> list[tuple[int, float]]:
        items = []
        for it in self._interactions:
            if it[0] != user_id: continue
            items.append((it[1], it[2]))
        return items
    
    def get_non_interacted_items(self, user_id) -> list[tuple[int, float]]:
        items = self.get_all_item_ids()
        interacted = set([x[0] for x in self.get_interacted_items(user_id)])
        items = items - interacted
        return list(items)
    
    def get_user_features_by_id(self, user_id) -> list[str]:
        u_idx = bisect.bisect_left(self._user_info, user_id, key=lambda x: x[0])
        u_found = (u_idx < len(self._user_info) and self._user_info[u_idx][0] == user_id)
        if not u_found:
            raise ValueError(f"User info not found: {user_id}.")
        u_info = self._user_info[u_idx]
        return u_info[1]
    
    def get_item_features_by_id(self, item_id) -> list[str]:
        i_idx = bisect.bisect_left(self._item_info, item_id, key=lambda x: x[0])
        i_found = (i_idx < len(self._item_info) and self._item_info[i_idx][0] == item_id)
        if not i_found:
            raise ValueError(f"Item info not found: {item_id}.")
        i_info = self._item_info[i_idx]
        return i_info[1]
    
    def get_all_item_ids(self) -> list[int]:
        items = set([x[0] for x in self._item_info])
        return items
    
    def get_all_user_ids(self) -> list[int]:
        users = set([x[0] for x in self._user_info])
        return users
    
    def __len__(self):
        return len(self._interactions)
    
    def __getitem__(self, idx: int):
        uid, iid, rating = self._interactions[idx]

        if self._cache_enabled and idx in self._cache:
            if self._verbose:
                logger.info(f"Cache found: {idx}.")
            u, i = self._cache[idx]
            return u, i, torch.tensor(rating, dtype=torch.float32)
        
        u_feat = self.get_user_features_by_id(uid)
        i_feat = self.get_item_features_by_id(iid)

        u, i = self.transform(
            user_id=uid,
            item_id=iid,
            user_features=u_feat,
            item_features=i_feat
        )

        if self._cache_enabled and idx not in self._cache:
            if self._verbose:
                logger.info(f"Cache new item: {idx}.")
            self._cache[idx] = (u, i)

        return u, i, torch.tensor(rating, dtype=torch.float32)