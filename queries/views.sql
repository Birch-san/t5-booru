create view blessed_files as
  select f.booru, f.fid
  from files f
  join tags t
    on t.booru = f.booru
   and t.fid = f.fid
  where t.tag_cat != 0
    and t.tag in ('hololive', 'touhou', 'neon_genesis_evangelion', 'mushoku_tensei', 're_zero_kara_hajimeru_isekai_seikatsu', 'fate_(series)', 'dota_(series)', 'dota_2', 'league_of_legends', 'kaguya-sama_wa_kokurasetai_~tensai-tachi_no_renai_zunousen~', 'tokyo_ghoul', 'higurashi_no_naku_koro_ni', 'persona', 'ore_no_imouto_ga_konna_ni_kawaii_wake_ga_nai', 'kyoto_animation', 'violet_evergarden', 'rwby', 'xenoblade_(series)', 'zombie_land_saga', 'eromanga_sensei', 'go-toubun_no_hanayome', 'sword_art_online', 'nier_(series)', 'chuunibyou_demo_koi_ga_shitai!', 'umineko_no_naku_koro_ni', 'kono_subarashii_sekai_ni_shukufuku_wo!', 'steins;gate', 'mahou_shoujo_madoka_magica', 'panty_&_stocking_with_garterbelt', 'boku_wa_tomodachi_ga_sukunai', 'toradora!', 'k-on!', 'hyouka', 'doki_doki_literature_club', 'clannad', 'air', 'kanon', 'angel_beats!', 'suzumiya_haruhi_no_yuuutsu', 'smol_ame', 'ousama_ranking', 'spice_and_wolf', 'spy_x_family', 'shingeki_no_kyojin', 'darling_in_the_franxx', 'nijisanji', 'lazulight', 'tate_no_yuusha_no_nariagari', 'toradora!', 'little_busters!', 'rewrite', 'ano_hi_mita_hana_no_namae_wo_bokutachi_wa_mada_shiranai.', 'koe_no_katachi', 'aria', 'hai_to_gensou_no_grimgar', 'seishun_buta_yarou', 'hori-san_to_miyamura-kun', 'kokoro_connect', 'vivy:_fluorite_eye''s_song', 'nagi_no_asukara', 'walfie', 'genshin_impact', 'honkai_impact_3rd', 'idolmaster', 'love_live!', 'yahari_ore_no_seishun_lovecome_wa_machigatteiru.', 'toaru_majutsu_no_index', 'magia_record:_mahou_shoujo_madoka_magica_gaiden', 'tamako_market', 'hibike!_euphonium', 'kantai_collection', 'ssss.gridman', 'fire_emblem:_three_houses') 
  group by f.booru, f.fid;
create view general_tag_count as
  select count(distinct (t.tag) ) 
  from blessed_files b
  join tags t
    on t.booru = b.booru
   and t.fid = b.fid
  where t.tag_cat = 0;
create view general_tag_count_popular as
  select count( * ) 
  from (
    select t.tag
    from blessed_files b
    join tags t
      on t.booru = b.booru
     and t.fid = b.fid
    where t.tag_cat = 0
    group by t.tag
    having count( * ) > 40
  );
create view general_tag_prevalence as
  select t.tag, count( * ) 
  from blessed_files b
  join tags t
    on t.booru = b.booru
    and t.fid = b.fid
  where t.tag_cat = 0
  group by t.tag;
create view general_tag_prevalence_popular as
  select t.tag, count( * ) 
  from blessed_files b
  join tags t
    on t.booru = b.booru
   and t.fid = b.fid
  where t.tag_cat = 0
  group by t.tag
  having count( * ) > 30;
create view general_tags as
  select distinct t.tag
  from blessed_files b
  join tags t
    on t.booru = b.booru
   and t.fid = b.fid
  where t.tag_cat = 0;
create view non_general_tag_count as
  select count(distinct (t.tag) ) 
  from blessed_files b
  join tags t
    on t.booru = b.booru
   and t.fid = b.fid
  where t.tag_cat != 0;
create view non_general_tag_count_popular as
  select count( * ) 
  from (
    select t.tag
    from blessed_files b
    join tags t
      on t.booru = b.booru
     and t.fid = b.fid
    where t.tag_cat != 0
    group by t.tag
    having count( * ) > 40
  );
create view non_general_tag_prevalence as
  select t.tag, count( * ) 
  from blessed_files b
  join tags t
    on t.booru = b.booru
   and t.fid = b.fid
  where t.tag_cat != 0
  group by t.tag;
create view non_general_tag_prevalence_popular as
  select t.tag, count( * ) 
  from blessed_files b
  join tags t
    on t.booru = b.booru and 
       t.fid = b.fid
  where t.tag_cat != 0
  group by t.tag
  having count( * ) > 30;
create view non_general_tags as
  select distinct t.tag
  from blessed_files b
  join tags t
    on t.booru = b.booru and 
       t.fid = b.fid
  where t.tag_cat != 0;