file = main.py

run:
	python $(file)
mem:
	compute-sanitizer --tool memcheck python $(file)
gdb:
	cuda-gdb python $(file)
ncu:
	ncu --set full python $(file) | tee output.log
ncu-gui:
	ncu --set full -f -o output python $(file)


