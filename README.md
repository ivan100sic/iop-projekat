# iop-projekat
Projekat iz predmeta Inteligentna obrada podataka

## Neke ideje

- Elementarni objekti: skalari, vektori, matrice
- Kontekst: CPU, GPU (preko OpenCL) ili mozda CUDA?
- Neki skup funkcija koje mogu da se primene nad elementarnim objektima: exp, log, sin, cos, sqrt, abs. (Zadrzati se na nekom manjem skupu jer ce da nam trebaju posebni kerneli za GPU za svaku, mada power copy-paste-a vrv ce da ucini svoje)

Primer:

- Napravim nekakav kontekst c koji je vezan za GPU, npr. `auto c = iop::context("GPU");`
- Napravim matricu i vektor (promenljive): `auto A = c.matrix(100, 100); auto b = c.vector(100)`;
- Napravim promenljivu koju vezujem za recimo sinus matricnog proizvoda: `auto v = iop::sin(A.dot(b));`

## Linkovi

- OpenCL specifikacija: https://www.khronos.org/registry/OpenCL/specs/opencl-1.1.pdf
- AMD OpenCL guide: http://developer.amd.com/wordpress/media/2013/12/AMD_OpenCL_Programming_User_Guide2.pdf
- GCN Arhitektura: http://developer.amd.com/wordpress/media/2013/12/AMD_GCN3_Instruction_Set_Architecture_rev1.1.pdf
- Za optimizaciju: https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/OpenCL_Best_Practices_Guide.pdf

## Note to self

- Optimizuj non stop. Uvek moze brze.
- Seti se sta te je Bane ucio u prvoj i drugoj godini faksa u OOP1 i OOP2.
- Seti se sta te Bane **nije** naucio - move constructor i move assignment.
- Nemoj da zaboravis da na kraju kod treba da bude koliko toliko citljiv, razmisli o mestima za refaktorisanje
- Nije lose da se napise sveobuhvatan `test.cpp` koji ce da testira svaku funkciju posebno
- Razmisli jos jednom o offline evaluaciji, ovo sigurno ima neke prednosti ali je drasticno zahtevnije za implementaciju
