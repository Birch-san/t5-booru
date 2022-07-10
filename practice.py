from torch.nn import Embedding, Module, CrossEntropyLoss

from typing import Tuple, TypeVar, Iterable, Iterator, Callable
from typing_extensions import TypeAlias
from enum import Enum, auto
from math import ceil
from torch import BoolTensor, LongTensor, FloatTensor, Tensor, sparse_coo_tensor, ones, zeros#, tensor
from itertools import chain, islice, count#, tee
from torch import bfloat16
from torch.optim import Optimizer, Adam#, SparseAdam


class Label(Enum):
  touhou = 0
  hololive = auto()
  marisa = auto()
  reimu = auto()
  youmu = auto()
  sakuya = auto()
  flandre = auto()
  reiuji = auto()
  reisen = auto()
  tewi = auto()
  patchouli = auto()
  aya = auto()
  pekora = auto()
  kronii = auto()
  gura = auto()
  suisei = auto()
  ame = auto()
  noel = auto()
  subaru = auto()
  kiara = auto()
  black_hair = auto()
  silver_hair = auto()
  blue_hair = auto()
  blonde_hair = auto()
  purple_hair = auto()
  orange_hair = auto()
  bunny_ears = auto()
  bird_person = auto()

vocab_size=len(Label)
# t5-small compressed 32100 vocab tokens into 512 dims
# there's plenty of range per bfloat16 to represent a variety of tokens
embedding_dim=ceil(512/32100 * vocab_size)

# class MultilabelEmbedding(Module):
#   embedding: Embedding
#   def __init__(self, embedding: Embedding) -> None:
#     super(MultilabelEmbedding, self).__init__()
#     self.embedding = embedding
  
#   def forward(self, batch_of_captions_tensor: BoolTensor) -> FloatTensor:
#     embeddings = self.embedding.forward(batch_of_captions_tensor)
#     return embeddings

embedding = Embedding(
  num_embeddings=vocab_size,
  embedding_dim=embedding_dim,
  sparse=True,
  dtype=bfloat16,
)

# model = MultilabelEmbedding(
#   embedding=embedding
# )
model = embedding

T = TypeVar('T')
U = TypeVar('U')
_Caption: TypeAlias = Tuple[Label, ...]
_Captions: TypeAlias = Tuple[_Caption, ...]

def make_row_indices(enumerated: Tuple[int, _Caption]) -> Tuple[int, ...]:
  (ix, labels) = enumerated
  return (ix,) * len(labels)

def flatten(captions: Iterable[Tuple[T, ...]]) -> Iterable[T]:
  return chain.from_iterable(captions)

def get_value(label: Label) -> int:
  return label.value

def captions_to_tensor(captions: _Captions) -> BoolTensor:
  row_indices: Tuple[int, ...] = tuple(flatten(map(make_row_indices, enumerate(captions))))
  labels: Tuple[int, ...] = tuple(map(get_value, flatten(captions)))

  indices_nominal: Tuple[Tuple[int, ...], Tuple[int, ...]] = (row_indices, labels)

  return sparse_coo_tensor(
    indices=LongTensor(indices_nominal),
    values=ones(len(row_indices), dtype=bool),
    size=(len(captions), vocab_size),
    dtype=bool).to_dense()

captions: _Captions = (
  (Label.touhou, Label.marisa, Label.blonde_hair),
  (Label.touhou, Label.reimu, Label.black_hair),
  (Label.touhou, Label.youmu, Label.silver_hair),
  (Label.touhou, Label.sakuya, Label.silver_hair),
  (Label.touhou, Label.flandre, Label.blonde_hair),
  (Label.touhou, Label.reiuji, Label.black_hair, Label.bird_person),
  (Label.touhou, Label.reisen, Label.purple_hair, Label.bunny_ears),
  (Label.touhou, Label.tewi, Label.black_hair, Label.bunny_ears),
  (Label.touhou, Label.patchouli, Label.purple_hair),
  (Label.touhou, Label.aya, Label.black_hair, Label.black_hair),
  (Label.hololive, Label.pekora, Label.blue_hair, Label.bunny_ears),
  (Label.hololive, Label.kronii, Label.blue_hair),
  (Label.hololive, Label.suisei, Label.blue_hair),
  (Label.hololive, Label.gura, Label.silver_hair),
  (Label.hololive, Label.noel, Label.silver_hair),
  (Label.hololive, Label.ame, Label.blonde_hair),
  (Label.hololive, Label.subaru, Label.black_hair, Label.bird_person),
  (Label.hololive, Label.kiara, Label.black_hair, Label.bird_person),
)

# def wraparound_iterator(get_iterator: Callable[[], Iterator[T]]) -> Iterator[Tuple[T, int]]:
#   for epoch in count(0):
#     iterator: Iterator[T] = get_iterator()
#     for caption in iterator:
#       yield (caption, epoch)

def wraparound_iterator(get_iterator: Callable[[], Iterator[T]]) -> Iterator[T]:
  while True:
    iterator: Iterator[T] = get_iterator()
    for caption in iterator:
      yield caption

def get_caption_iterator() -> Iterator[_Caption]:
  return iter(captions)

def batches_of(iterator: Iterable[T], batch_size: int) -> Iterator[Tuple[T, ...]]:
  while True:
    yield tuple(islice(iterator, batch_size))

_EpochZipped: TypeAlias = Tuple[int, T]

def zip_epoch(iterator: Iterator[T]) -> Iterator[_EpochZipped[T]]:
  for epoch in count(0):
    for element in iter(iterator):
      yield (epoch, element)

# def map_epoch_zipped(zipped: Iterable[_EpochZipped[T]], operate: Callable[[Iterable[T]], Iterable[U]]) -> Iterator[_EpochZipped[U]]:
#   for (epoch, value) in zipped:
#     yield (epoch, operate(value))

def map_batched_epoch_zipped(zipped_batches: Iterable[Tuple[_EpochZipped[T], ...]], operate: Callable[[Tuple[T, ...]], U]) -> Iterator[_EpochZipped[U]]:
  for zipped_batch in zipped_batches:
    (epoch, *_), *_ = zipped_batch
    yield (epoch, operate(tuple(map(lambda tup: tup[1], zipped_batch))))

batches: Iterable[_EpochZipped[BoolTensor]] = map_batched_epoch_zipped(batches_of(zip_epoch(captions), 2), captions_to_tensor)

# batches: Iterable[int, BoolTensor] = map(captions_to_tensor, batches_of(get_caption_iterator, 2))

class Trainer:
  model: Module
  epochs: int
  batches: Iterable[_EpochZipped[BoolTensor]]
  opt: Optimizer
  loss_fn: Callable[[Tensor, Tensor], Tensor]
  true_value: FloatTensor
  def __init__(
    self,
    model: Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    opt: Optimizer,
    batches: Iterable[_EpochZipped[BoolTensor]],
    epochs: int,
    true_value: FloatTensor
  ) -> None:
    self.model = model
    self.epochs = epochs
    self.batches = batches
    self.opt = opt
    self.loss_fn = loss_fn
    self.true_value = true_value

  def train(self):
    for step, (epoch, batch_tensor) in enumerate(batches):
      if epoch >= self.epochs:
        return
      prediction = self.model(batch_tensor)
      loss: Tensor = self.loss_fn(prediction, self.true_value)
      # if step % 100 == 0:
      print(step, loss.item())
      self.opt.zero_grad()
      loss.backward()
      self.opt.step()

learning_rate = 1e-3
# opt = SparseAdam(model.parameters(), lr=learning_rate)
opt = Adam(model.parameters(), lr=learning_rate)
loss_fn = CrossEntropyLoss()
true_value: FloatTensor = zeros(size=(embedding_dim,), dtype=bfloat16)
trainer = Trainer(model=model, batches=batches, epochs=1, opt=opt, loss_fn=loss_fn, true_value=true_value)
trainer.train()