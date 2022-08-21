from boorupiece_simple.boorupiece import BooruPiece

tokenizer = BooruPiece()

# I am skipping `tokenizer.regularize_label()` because I already know it's lowercase

# https://danbooru.donmai.us/wiki_pages/help:tags
# _(cosplay) must be preceded by a name label, so we split into just two tokens
# ('nero_claudius_(fate)', 'cosplay')
tokens = tuple(tokenizer.tokenize_label('nero_claudius_(fate)_(cosplay)'))
print(tokens)

# <unk> token used when unknown
# ('<unk>', 'cosplay')
tokens = tuple(tokenizer.tokenize_label('nero_claudius_(swimsuit_caster)_(fate)_(cosplay)'))
print(tokens)

# for qualifiers other than cosplay, we accept multiple
# ('orange', 'fruit', 'meme')
tokens = tuple(tokenizer.tokenize_label('orange_(fruit)_(meme)'))
print(tokens)

# for qualifiers other than cosplay, we split what precedes it
# ('coffee', 'maker', 'object')
tokens = tuple(tokenizer.tokenize_label('coffee_maker_(object)'))
print(tokens)

# one consequence of our splitting is that we miss the opportunity to whole-label match on 'dragon_ball'
# we could fix that, but I don't think it's so bad to have a separate embedding
# for the concept which is a type of ball, versus the concept which is a franchise
# ('dragon', 'ball', 'object')
tokens = tuple(tokenizer.tokenize_label('dragon_ball_(object)'))
print(tokens)

# we first check if the vocab has a whole-label match (like we do for names)
# but otherwise we split on hyphens and underscores
# ('looking', 'at', 'viewer')
tokens = tuple(tokenizer.tokenize_label('looking_at_viewer'))
print(tokens)

# parenthesized qualifiers do not undergo splitting
# ('jinx', 'league_of_legends')
tokens = tuple(tokenizer.tokenize_label('jinx_(league_of_legends)'))
print(tokens)

# we only attempt splitting if label length > 4, because anything smaller is likely to be kaomoji
# ('=_=')
tokens = tuple(tokenizer.tokenize_label('=_='))
print(tokens)

# we split on hyphens
# ('v', 'shaped', 'eyebrows')
tokens = tuple(tokenizer.tokenize_label('v-shaped_eyebrows'))
print(tokens)