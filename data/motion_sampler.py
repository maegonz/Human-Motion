import random as rd
from torch.utils.data import Sampler
from collections import defaultdict

rd.seed(42)

class MotionSampler(Sampler):
    def __init__(self, frames, batch_size=32, bucket_size=32, shuffle=True):
        """
        Sampler that groups sequences of similar frames into the same batch.

        Parameters
        ----------
        frames : list of int
            List containing the frames of each sequence in the dataset.
        batch_size : int
            Number of samples per batch. Defaults to 32.
        shuffle : bool, optional
            Whether to shuffle the batches, by default True.
        """
        self.frames = frames
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        # max_frames =150

        # Group indices by frames
        buckets = defaultdict(list)
        for id, frame in enumerate(frames):
            bucket_id = frame // self.bucket_size
            buckets[bucket_id].append(id)
        
        # print(f"Number of buckets created: {len(buckets)}")
        # print(f"Bucket ids: {sorted(buckets.keys())}")
        # print(f"Bucket sizes: {[len(v) for v in buckets.values()]}")
        # print(f"Bucket examples: {list(buckets.values())[:5]}")

        # Create batches
        self.batches = []
        for bucket in buckets.values():
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                self.batches.append(batch)
        
        # print(f"Total number of batches created: {len(self.batches)}")
        # print(f"Batch sizes: {[len(b) for b in self.batches]}")

        if self.shuffle:
            rd.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)
    
    def __iter__(self):
        yield from self.batches

# import random as rd
# example = MotionSampler(frames=[rd.randint(10, 150) for _ in range(146)], batch_size=16, shuffle=True)

# class MotionSampler(Sampler):
#     def __init__(
#         self,
#         frames,
#         batch_size,
#         bucket_size=32,
#         max_frames=150,
#         shuffle=True
#     ):
#         self.batch_size = batch_size
#         self.shuffle = shuffle

#         # Clip long sequences (IMPORTANT)
#         frames = [min(f, max_frames) for f in frames]

#         # Bucket by length ranges
#         buckets = defaultdict(list)
#         for idx, f in enumerate(frames):
#             bucket_id = f // bucket_size
#             buckets[bucket_id].append(idx)

#         self.batches = []
#         for bucket in buckets.values():
#             if shuffle:
#                 random.shuffle(bucket)
#             for i in range(0, len(bucket), batch_size):
#                 batch = bucket[i:i + batch_size]
#                 if len(batch) == batch_size:
#                     self.batches.append(batch)

#         if shuffle:
#             random.shuffle(self.batches)

#     def __len__(self):
#         return len(self.batches)

#     def __iter__(self):
#         yield from self.batches
  