lang_codes="nl el no sl bg en pl es ca et hu pt sv hr fi id ro tr cs fr it ru uk da de mk sk vi"
echo "STARTING"
for lang_code in $lang_codes
do
	echo "PROCESSING -> $lang_code"
	python counting_words_per_concept.py --voc_size 200000 --lang_code $lang_code
done
echo "DONE"
