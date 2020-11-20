gcc_options = -std=c++17 -Wall --pedantic-error -O3
FILENAME = comp_phys_report_202020190.cpp
# FILENAME = 2D_Iging.cpp

a.out : $(FILENAME)
	g++ $(gcc_options) -o $@ $<

run : a.out
	./a.out

clean :
	rm -f ./a.out

animation :
	python3 ../analytic_codes/animation.py
	python3 ../analytic_codes/animation_all.py
	cp ../figs/animation.mp4 ../data/steady
	cp ../figs/animation_all.mp4 ../data/steady

mv_nohup :
	cp nohup.out ../data/steady

rm_time_data :
	rm -r -f ../data/J/xJ/*
	rm -r -f ../data/J/rJ/*

.PHONY : run clean animation rm_rime_data mkdir_data