CREATE VIEW blessed_files AS
    SELECT f.BOORU,
           f.FID
      FROM files f
           JOIN
           tags t ON t.BOORU = f.BOORU AND 
                     t.FID = f.FID
     WHERE t.TAG_CAT != 0 AND 
           t.TAG IN ('hololive', 'touhou', 'neon_genesis_evangelion', 'mushoku_tensei', 're_zero_kara_hajimeru_isekai_seikatsu', 'fate_(series)', 'dota_(series)', 'dota_2', 'league_of_legends', 'kaguya-sama_wa_kokurasetai_~tensai-tachi_no_renai_zunousen~', 'tokyo_ghoul', 'higurashi_no_naku_koro_ni', 'persona', 'ore_no_imouto_ga_konna_ni_kawaii_wake_ga_nai', 'kyoto_animation', 'violet_evergarden', 'rwby', 'xenoblade_(series)', 'zombie_land_saga', 'eromanga_sensei', 'go-toubun_no_hanayome', 'sword_art_online', 'nier_(series)', 'chuunibyou_demo_koi_ga_shitai!', 'umineko_no_naku_koro_ni', 'kono_subarashii_sekai_ni_shukufuku_wo!', 'steins;gate', 'mahou_shoujo_madoka_magica', 'panty_&_stocking_with_garterbelt', 'boku_wa_tomodachi_ga_sukunai', 'toradora!', 'k-on!', 'hyouka', 'doki_doki_literature_club', 'clannad', 'air', 'kanon', 'angel_beats!', 'suzumiya_haruhi_no_yuuutsu', 'smol_ame', 'ousama_ranking', 'spice_and_wolf', 'spy_x_family', 'shingeki_no_kyojin', 'darling_in_the_franxx', 'nijisanji', 'lazulight', 'tate_no_yuusha_no_nariagari', 'toradora!', 'little_busters!', 'rewrite', 'ano_hi_mita_hana_no_namae_wo_bokutachi_wa_mada_shiranai.', 'koe_no_katachi', 'aria', 'hai_to_gensou_no_grimgar', 'seishun_buta_yarou', 'hori-san_to_miyamura-kun', 'kokoro_connect', 'vivy:_fluorite_eye''s_song', 'nagi_no_asukara', 'walfie', 'genshin_impact', 'honkai_impact_3rd', 'idolmaster', 'love_live!', 'yahari_ore_no_seishun_lovecome_wa_machigatteiru.', 'toaru_majutsu_no_index', 'magia_record:_mahou_shoujo_madoka_magica_gaiden', 'tamako_market', 'hibike!_euphonium', 'kantai_collection', 'ssss.gridman', 'fire_emblem:_three_houses') 
     GROUP BY f.BOORU,
              f.FID;
CREATE VIEW general_tag_count AS
    SELECT count(DISTINCT (t.TAG) ) 
      FROM blessed_files b
           JOIN
           tags t ON t.BOORU = b.BOORU AND 
                     t.FID = b.FID
     WHERE t.TAG_CAT = 0;
CREATE VIEW general_tag_count_popular AS
    SELECT count( * ) 
      FROM (
               SELECT t.TAG
                 FROM blessed_files b
                      JOIN
                      tags t ON t.BOORU = b.BOORU AND 
                                t.FID = b.FID
                WHERE t.TAG_CAT = 0
                GROUP BY t.TAG
               HAVING count( * ) > 40
           );
CREATE VIEW general_tag_prevalence AS
    SELECT t.TAG,
           count( * ) 
      FROM blessed_files b
           JOIN
           tags t ON t.BOORU = b.BOORU AND 
                     t.FID = b.FID
     WHERE t.TAG_CAT = 0
     GROUP BY t.TAG;
CREATE VIEW general_tag_prevalence_popular AS
    SELECT t.TAG,
           count( * ) 
      FROM blessed_files b
           JOIN
           tags t ON t.BOORU = b.BOORU AND 
                     t.FID = b.FID
     WHERE t.TAG_CAT = 0
     GROUP BY t.TAG
    HAVING count( * ) > 30;
CREATE VIEW general_tags AS
    SELECT DISTINCT t.TAG
      FROM blessed_files b
           JOIN
           tags t ON t.BOORU = b.BOORU AND 
                     t.FID = b.FID
     WHERE t.TAG_CAT = 0;
CREATE VIEW non_general_tag_count AS
    SELECT count(DISTINCT (t.TAG) ) 
      FROM blessed_files b
           JOIN
           tags t ON t.BOORU = b.BOORU AND 
                     t.FID = b.FID
     WHERE t.TAG_CAT != 0;
CREATE VIEW non_general_tag_count_popular AS
    SELECT count( * ) 
      FROM (
               SELECT t.TAG
                 FROM blessed_files b
                      JOIN
                      tags t ON t.BOORU = b.BOORU AND 
                                t.FID = b.FID
                WHERE t.TAG_CAT != 0
                GROUP BY t.TAG
               HAVING count( * ) > 40
           );
CREATE VIEW non_general_tag_prevalence AS
    SELECT t.TAG,
           count( * ) 
      FROM blessed_files b
           JOIN
           tags t ON t.BOORU = b.BOORU AND 
                     t.FID = b.FID
     WHERE t.TAG_CAT != 0
     GROUP BY t.TAG;
CREATE VIEW non_general_tag_prevalence_popular AS
    SELECT t.TAG,
           count( * ) 
      FROM blessed_files b
           JOIN
           tags t ON t.BOORU = b.BOORU AND 
                     t.FID = b.FID
     WHERE t.TAG_CAT != 0
     GROUP BY t.TAG
    HAVING count( * ) > 30;
CREATE VIEW non_general_tags AS
    SELECT DISTINCT t.TAG
      FROM blessed_files b
           JOIN
           tags t ON t.BOORU = b.BOORU AND 
                     t.FID = b.FID
     WHERE t.TAG_CAT != 0;