
help:
	@echo "makefile of creation of mzn problem files.\n\
		all: inference.mzn, improvement.mzn, phi.mzn\n\
		clean: remove *.mzn files"

all: inference improvement phi utility

inference:
	cat problem.pzn inference.pzn > inference.mzn

improvement:
	cat problem.pzn improvement.pzn > improvement.mzn

phi:
	cat problem.pzn phi.pzn > phi.mzn

utility:
	cat problem.pzn utility.pzn > utility.mzn
clean:
	rm -f inference.mzn
	rm -f improvement.mzn
	rm -f phi.mzn
	rm -f utility.mzn
	rm -f *.fzn
	rm -f *.ozn


