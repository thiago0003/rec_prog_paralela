ftime=TimeExec.csv 
exe=nbody

if [ -z "$1" ]; then
	echo "Usage: $0 <size_matrix>"
else

	if [ ! -f "$ftime" ]; then
		# Create file and fill with header
		touch "$ftime"
		echo "platform,nbodies,execution_time_s" > "$ftime"
	fi

	nbodies="$1"
    make input nbodies=$nbodies

	#executa 5 vezes cada um dos algoritmos

	for val in {seq 0 4}; do 
		echo "Execucao nro - $val"
		for plt in {"c","omp"}; do
			make nbody_$plt
			env time --append --format "CPU-$plt, $nbodies,%e" \
					--output "$ftime" ./$exe-$plt < input.txt ;
			
		done
		
		for ver in {0,1}; do
			make nbody_gpu version=$ver
			env time --append --format "GPU-CUDA_v$ver, $nbodies,%e" \
					--output "$ftime" ./$exe-v$ver < input.txt ;
		
		done
	done

	python3 include/graphic.py
    
    # make clean


fi

	