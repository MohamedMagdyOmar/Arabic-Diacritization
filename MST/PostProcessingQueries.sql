SET SQL_SAFE_UPDATES = 0;
UPDATE diaconehotencoding SET DiacritizedCharacterOneHotEncoding = REPLACE(DiacritizedCharacterOneHotEncoding, ' ', '');
UPDATE diaconehotencoding SET DiacritizedCharacterOneHotEncoding = REPLACE(DiacritizedCharacterOneHotEncoding, '[', '');
UPDATE diaconehotencoding SET DiacritizedCharacterOneHotEncoding = REPLACE(DiacritizedCharacterOneHotEncoding, ']', '');
UPDATE diaconehotencoding SET DiacritizedCharacterOneHotEncoding = REPLACE(DiacritizedCharacterOneHotEncoding, '\n', '');


SET SQL_SAFE_UPDATES = 0;
UPDATE undiaconehotencoding SET UnDiacritizedCharacterOneHotEncoding = REPLACE(UnDiacritizedCharacterOneHotEncoding, ' ', '');
UPDATE undiaconehotencoding SET UnDiacritizedCharacterOneHotEncoding = REPLACE(UnDiacritizedCharacterOneHotEncoding, '[', '');
UPDATE undiaconehotencoding SET UnDiacritizedCharacterOneHotEncoding = REPLACE(UnDiacritizedCharacterOneHotEncoding, ']', '');
UPDATE undiaconehotencoding SET UnDiacritizedCharacterOneHotEncoding = REPLACE(UnDiacritizedCharacterOneHotEncoding, '\n', '');

SET SQL_SAFE_UPDATES = 0;
UPDATE distinctdiacritics SET encoding = REPLACE(encoding, ' ', '');
UPDATE distinctdiacritics SET encoding = REPLACE(encoding, '[', '');
UPDATE distinctdiacritics SET encoding = REPLACE(encoding, ']', '');

SET SQL_SAFE_UPDATES = 0;
UPDATE diacritics_and_undiacritized_letter_one_hot_encoding SET OneHotEncoding = REPLACE(OneHotEncoding, ' ', '');
UPDATE diacritics_and_undiacritized_letter_one_hot_encoding SET OneHotEncoding = REPLACE(OneHotEncoding, '[', '');
UPDATE diacritics_and_undiacritized_letter_one_hot_encoding SET OneHotEncoding = REPLACE(OneHotEncoding, ']', '');


-- reset auto increment column
SET SQL_SAFE_UPDATES = 0;
SET  @num := 0;
UPDATE distinctdiacritics SET id= @num := (@num+1);
ALTER TABLE distinctdiacritics AUTO_INCREMENT =1;


SET SQL_SAFE_UPDATES = 0;
SET  @num := 0;
UPDATE parseddocument SET idCharacterNumber= @num := (@num+1);
ALTER TABLE parseddocument AUTO_INCREMENT =1;

SET SQL_SAFE_UPDATES = 0;
SET  @num := 0;
UPDATE encodedwords SET id= @num := (@num+1);
ALTER TABLE encodedwords AUTO_INCREMENT =1;

SET SQL_SAFE_UPDATES = 0;
SET  @num := 0;
UPDATE ListOfWordsAndSentencesInEachDoc SET idWord= @num := (@num+1);
ALTER TABLE ListOfWordsAndSentencesInEachDoc AUTO_INCREMENT =1;

SET SQL_SAFE_UPDATES = 0;
SET  @num := 0;
UPDATE diacritics_and_undiacritized_letter_one_hot_encoding SET id= @num := (@num+1);
ALTER TABLE diacritics_and_undiacritized_letter_one_hot_encoding AUTO_INCREMENT =1;


SET SQL_SAFE_UPDATES = 0;
SET  @num := 0;
UPDATE undiaconehotencoding SET idUnDiacritizedCharacter= @num := (@num+1);
ALTER TABLE undiaconehotencoding AUTO_INCREMENT =1;

SET SQL_SAFE_UPDATES = 0;
SET  @num := 0;
UPDATE diaconehotencoding SET idDiacritizedCharacter= @num := (@num+1);
ALTER TABLE diaconehotencoding AUTO_INCREMENT =1;

SET SQL_SAFE_UPDATES = 0;
SET  @num := 0;
UPDATE distinctdiacritics SET id= @num := (@num+1);
ALTER TABLE distinctdiacritics AUTO_INCREMENT =1;

SET SQL_SAFE_UPDATES = 0;
SET  @num := 0;
UPDATE dictionary SET idDictionary= @num := (@num+1);
ALTER TABLE dictionary AUTO_INCREMENT =1;

SET SQL_SAFE_UPDATES = 0;
SET  @num := 0;
UPDATE arabic_letters_with_diacritics SET id= @num := (@num+1);
ALTER TABLE arabic_letters_with_diacritics AUTO_INCREMENT =1;

SET SQL_SAFE_UPDATES = 0;
SET  @num := 0;
UPDATE arabic_letters_without_diacritics SET id= @num := (@num+1);
ALTER TABLE arabic_letters_without_diacritics AUTO_INCREMENT =1;

-- the following "update" commands is for "atb3"
-- update parseddocument set Diacritics='ٰ';file:/D:/Repos/MST/MST/Encoding/CreatingOneHotNew.py
UPDATE parseddocument SET Diacritics = REPLACE(Diacritics,'ٰ','');
UPDATE parseddocument SET Diacritics = REPLACE(Diacritics,'ًً','ً');
UPDATE parseddocument SET DiacritizedCharacter = REPLACE(DiacritizedCharacter,'اًً','اً');

-- Dictionary Table Creation
insert into dictionary (DiacritizedWord,UnDiacritizedWord)
(select word, UnDiacritizedWord from parseddocument where   LetterType='training' group by word order by UnDiacritizedWord asc);

-- presence of shadda only mistake
select  distinct idCharacterNumber, Diacritics from parseddocument where diacritics = 'ّ';

UPDATE parseddocument
SET Diacritics = 'َّ' 
WHERE Diacritics = 'ّ' ;

SET SQL_SAFE_UPDATES = 0;
UPDATE parseddocument
SET DiacritizedCharacter = 'َّ' 
WHERE DiacritizedCharacter = 'ّ' ;

SET SQL_SAFE_UPDATES = 0;
UPDATE parseddocument
SET DiacritizedCharacter = 'تَّ'
WHERE DiacritizedCharacter = 'تّ' 


