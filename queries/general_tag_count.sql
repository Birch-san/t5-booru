select count(distinct(t.TAG))
from blessed_files b
join tags t
  on t.BOORU = b.BOORU
 and t.FID = b.FID
where t.TAG_CAT = 0