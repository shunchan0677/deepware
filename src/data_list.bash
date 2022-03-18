for file in `\find /media/brainiv/PioMeidai -name '_*.bag'`; do
  echo $file
  output=${file%.*}
  input=$file
  echo $output
  echo $input
  python carla_data_maker.py $output $input

done
