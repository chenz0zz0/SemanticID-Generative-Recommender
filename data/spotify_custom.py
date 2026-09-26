import os
import pandas as pd
import torch
import numpy as np
from .preprocessing import PreprocessingMixin
from torch_geometric.data import HeteroData, InMemoryDataset
from typing import Optional, Callable, List

class SpotifyCustom(InMemoryDataset, PreprocessingMixin):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        split: str = None,  # <-- Add this
        **kwargs            # <-- And this
    ) -> None:
        super().__init__(root, transform, pre_transform, force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return ['all.csv']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def _generate_playlist_sequences(self, df):
        """
        Groups tracks by (user_id, playlist_name) as sessions.
        For each session, randomly assigns one track to val and one to test, rest to train.
        Returns three lists of (user_idx, playlist_name, track_idx) for train/val/test.
        """
        rng = np.random.default_rng(42)
        train, val, test = [], [], []

        # Group by user and playlist
        for (user_id, playlist_name), group in df.groupby(['user_id', 'playlist_name']):
            indices = group.index.to_numpy()
            if len(indices) < 3:
                # Not enough tracks for all splits, skip or assign all to train
                for idx in indices:
                    train.append((df.at[idx, 'user_idx'], playlist_name, df.at[idx, 'track_idx']))
                continue
            chosen = rng.choice(indices, size=2, replace=False)
            val_idx, test_idx = chosen
            for idx in indices:
                entry = (df.at[idx, 'user_idx'], playlist_name, df.at[idx, 'track_idx'])
                if idx == val_idx:
                    val.append(entry)
                elif idx == test_idx:
                    test.append(entry)
                else:
                    train.append(entry)
        return train, val, test

    def process(self, max_seq_len=100) -> None:
        data = HeteroData()
        df = pd.read_csv(os.path.join(self.raw_dir, 'all.csv'), sep=',', index_col=0)
        print("Columns:", df.columns)  # Debugging

        # Map user and track IDs to integer indices
        user2id = {uid: i for i, uid in enumerate(df['user_id'].unique())}
        track2id = {tid: i for i, tid in enumerate(df['id'].unique())}

        df['user_idx'] = df['user_id'].map(user2id)
        df['track_idx'] = df['id'].map(track2id)

        # Extract item (track) features
        item_features = [
            'valence', 'year', 'acousticness', 'danceability', 'duration_ms',
            'energy', 'explicit', 'instrumentalness', 'key', 'liveness',
            'loudness', 'mode', 'popularity', 'speechiness', 'tempo'
        ]
        item_df = df.drop_duplicates('track_idx')
        x_item = torch.tensor(item_df[item_features].values, dtype=torch.float)
        data['item'].x = x_item

        # Encode text features (track name + artist)
        text_inputs = (
            item_df['name'] + ' by ' + item_df['artist'] +
            ', year: ' + item_df['year'].astype(str) +
            ', valence: ' + item_df['valence'].astype(str) +
            ', danceability: ' + item_df['danceability'].astype(str) +
            ', energy: ' + item_df['energy'].astype(str) +
            ', popularity: ' + item_df['popularity'].astype(str)
            # Add more features as needed
        ).tolist()
        data['item'].text = self._encode_text_feature(text_inputs)

        # Generate playlist sequences and splits
        train_seqs, val_seqs, test_seqs = self._generate_playlist_sequences(df)

        # Mark items as train or not
        train_track_idxs = set(idx for _, _, idx in train_seqs)
        is_train = item_df['track_idx'].apply(lambda idx: idx in train_track_idxs).values
        data['item']['is_train'] = torch.tensor(is_train, dtype=torch.bool)

        # Extract user features if desired (or use dummy features)
        user_count = len(user2id)
        data['user'].x = torch.zeros((user_count, 1))  # Dummy feature

        # Build user-item interaction edges
        src = torch.tensor(df['user_idx'].values, dtype=torch.long)
        dst = torch.tensor(df['track_idx'].values, dtype=torch.long)
        data['user', 'rates', 'item'].edge_index = torch.stack([src, dst])

        # Optionally, add rating or timestamp if available
        # data['user', 'rates', 'item'].rating = torch.ones_like(src)

        # You can store these as attributes or add to data as needed for your model
        data['playlist'] = {
            'train': train_seqs,
            'val': val_seqs,
            'test': test_seqs,
        }

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])