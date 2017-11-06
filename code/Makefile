SUBDIRS= nvidia osu_allreduce baidu_allreduce

subdirs: $(SUBDIRS)

$(SUBDIRS):
	mkdir -p bin
	$(MAKE) -C $@
	cp $@/bin/* bin
	rm -rf $@/bin

.PHONY: subdirs $(SUBDIRS)

clean: 
	rm -rf bin
