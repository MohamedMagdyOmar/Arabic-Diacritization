-- General Selection
select * from parseddocument;
select * from encodedwords;
select * from listofwordsandsentencesineachdoc;
select * from undiaconehotencoding;
select * from diacritics_and_undiacritized_letter_one_hot_encoding;
select * from diaconehotencoding;
select * from distinctdiacritics;
select * from alldiacriticsinalldocuments;
select * from dictionary;
select * from arabic_letters_without_diacritics;
select * from arabic_letters_with_diacritics;
select * from arabic_diacritics;
select * from arabic_letters_and_diacritics;
select * from parseddocument where Diacritics = 'َّ' and location = 'middle';

SELECT m1.*
FROM parseddocument m1 LEFT JOIN parseddocument m2
 ON (m1.Word = m2.Word AND m1.idCharacterNumber < m2.idCharacterNumber)
WHERE m2.idCharacterNumber IS NULL;


-- Conditional Selection

-- Parsed Table
select * from parseddocument where   LetterType='training' and SentenceNumber=1646 ;
select * from parseddocument where   LetterType='training' and Diacritics = '' ;
select * from parseddocument where   LetterType='validation';
select  * from parseddocument where LetterType='testing' and DiacritizedCharacter = 'تّ' and SentenceNumber=1 and DocName = 'ANN20021015.0101.txt';
select * from parseddocument where LetterType='testing' and SentenceNumber = 1;

-- this represent alef when it is found above character, this is found only in atb3
select  * from parseddocument where Diacritics= 'ٰ';
select  * from parseddocument where UnDiacritizedCharacter= 'ـ';
select  * from parseddocument where Diacritics= 'ًً';
select distinct SentenceNumber from parseddocument where LetterType = 'testing';

select *from parseddocument where location = 'last' and DiacritizedCharacter = 'آ';
-- shadda error in atb3
select * from parseddocument where Diacritics = 'ّ';

-- fatha correction words and letters in parsed table
select distinct word from parseddocument where UnDiacritizedCharacter = (select arabic_letter from arabic_letters_without_diacritics where id = 7);
select distinct word, UnDiacritizedCharacter from parseddocument where UnDiacritizedCharacter = (select arabic_letter from arabic_letters_without_diacritics where id = 2);
select distinct word, UnDiacritizedCharacter from parseddocument where UnDiacritizedCharacter = (select arabic_letter from arabic_letters_without_diacritics where id = 34);


-- UnDiacOneHotEncoding table
select * from UnDiacOneHotEncoding where UnDiacritizedCharacter='' or UnDiacritizedCharacter='.';


-- diaconehotencoding table
select * from diaconehotencoding order by DiacritizedCharacter asc;


-- ListOfWordsAndSentencesInEachDoc table
select * from ListOfWordsAndSentencesInEachDoc where word = '' ;
select * from ListOfWordsAndSentencesInEachDoc where word != 'space' and word != 'bos' and word != 'eos' ;


-- dictionary table
select DiacritizedWord from dictionary where  UnDiacritizedWord = 'إنا';
select * from dictionary where DiacritizedWord ='فَتَمَنَّوُا';
select * from dictionary where DiacritizedWord ='وَرَأَوُا';
select * from dictionary where DiacritizedWord ='يَرَوُا';
select * from dictionary where DiacritizedWord ='تَحَرَّوْا';
select * from dictionary where DiacritizedWord ='اهْتَدَوْا';
select * from dictionary where DiacritizedWord ='تَحَرَّوْا';
select * from dictionary where DiacritizedWord ='كُفُوًا';
select * from dictionary where DiacritizedWord ='هُزُوًا';
select * from dictionary where unDiacritizedWord ='ربه';

CREATE TABLE new_foo LIKE parseddocument;
RENAME TABLE parseddocument TO old_foo, new_foo TO parseddocument;
DROP TABLE old_foo;

CREATE TABLE new_foo LIKE encodedwords;
RENAME TABLE encodedwords TO old_foo, new_foo TO encodedwords;
DROP TABLE old_foo;

CREATE TABLE new_foo LIKE ListOfWordsAndSentencesInEachDoc;
RENAME TABLE ListOfWordsAndSentencesInEachDoc TO old_foo, new_foo TO ListOfWordsAndSentencesInEachDoc;
DROP TABLE old_foo;
