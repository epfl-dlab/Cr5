if [ $# -eq 0 ]
  then
  	echo "Usage: ./create_folder_structure.sh data_directory_path"
    echo "To work with the default setting. Pass './..'"
    exit 1
fi

HOME=$1

mkdir $HOME/data
mkdir $HOME/data/embeddings_dumps
mkdir $HOME/data/target_concepts
mkdir $HOME/data/logs
mkdir $HOME/data/results_dumps
mkdir $HOME/data/training_concepts
mkdir $HOME/data/training_datasets
mkdir $HOME/data/validation_concepts
mkdir $HOME/data/validation_datasets
mkdir $HOME/data/voc_mappings
mkdir $HOME/data/word_counts