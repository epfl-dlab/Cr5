if [ $# -le 1 ]
  then
    echo "Usage: ./get_baseline_embeddings.sh lang_code data_directory_path"
    echo "To work with the default data directory. Pass './..'"
    exit 1
fi

url=`printf https://s3.amazonaws.com/arrival/embeddings/wiki.multi.%s.vec "$1"`
embeddings_dumps_file_path=`printf %s/data/embeddings_dumps/ $2`
wget --directory-prefix=$embeddings_dumps_file_path $url