15 min Präsentation:
- Start: Vergleichsfolie - normale Linse (effective phase durch unterschiedliche Dicke) und Grin Linse (phase durch n unterschied) --> nur im Wellenoptik Bild
- Einleitung (Grin linsen zeigen, Herstellungbedingungen)
- Einfluss von Belichtungsdauer und Laserleistung auf Linsen
- Compositional Measurements through transmission spectra
	- Setup (WiTec)
	- GA + Benchmarks
	- Results for two lenses (2W and 3W) (zeige vielleicht nur reinen Fitting Ansatz und Mixing in den Anhang)
- 

## Introduction:
1. General introduction into the topic/ papers:
	- Historically (Maxwell Fisheye lens/ Luneburg lens) spherical symmetric GRIN lens, with largest refractive index in the center (https://en.wikipedia.org/wiki/Luneburg_lens):
		- collimates light that is focused on one side of the sphere to the other side![[luneburg-lens-from-Wikipedia.png]] (Source Wikipedia Luneburg-lens)
		- first mathematical discription in 1854 (["Solutions of problems (prob. 3, vol. VIII. p. 188)"](https://books.google.com/books?id=-bI5AAAAMAAJ&pg=PA9). _The Cambridge and Dublin Mathematical Journal_. **9**. Macmillan: 9–11. 1854.)
	- All eye lenses in nature utilize the concept of GRIN lenses ([https://doi.org/10.1016/j.preteyeres.2012.03.001](https://doi.org/10.1016/j.preteyeres.2012.03.001 "Persistent link using digital object identifier"))
		- Index gradient increase between peripheral and center of the lens
		- Gradient and magnitude of GRIN lenses strongly vary from species to species
	- Biomedical applications (Endoscopes) - Source: (https://www.nature.com/articles/s41377-021-00648-w)
		- State of the art approach to analyze tissue in vivo 
		- Analysis of tissue structures using specialized GRIN lens endoscopy lens assembly 
		- Very compact in size, tuneable for exect purpose
	- Used to built specialized optical fibers, that do not rely on total internal refraction, but focus the light back to the center by means of Gradient Refractive index inside the fiber. Resulting in the same propagation velocities for all modes (Duncan T. Moore, "Gradient-index optics: a review," Appl. Opt. **19**, 1035-1038 (1980))
2. Our approach:
	- Develop a GRIN lens from mixture of two oxides
	- 
## Theoretical Background
Split into two parts, the theory of **radial GRIN** lenses and the theory of TMM and GA
1. GRIN lenses (wave-front bending and how they compare to conventional lenses)
	- Workingprinciple of conventional lenses (varying material thickness + **constant refractive index** + Snells Law at interfaces causes deflection of beam and creates focus) - **Ray Optics**
	- Waveoptics picture: (different amount of material thickness in conventional lens causes different amount of phase delay of incoming plane waves, which in turn creates circular/ bent wavefronts resulting in focus)
	- GRIN lens: Vary the refractive index to create phase delay, not the amount of material, to create curvature in wavefronts
2. Transfer Matrix Formalism
3. Genetic Algorithm

## Experimental Schemes
1. Sample Preparation:
	- Creation of copper films: Calibration with quartz microbalance, problem of determining suited film thickness (transmission amplitude vs. lens refractive strength)
	- Laser annealing: Characterisation of the laser source (Power series with bolometer), Setup sketch for in-situ laser annealing, mention that without O2 in chamber no lenses did form (i.e. no optical difference to bare Cu), Sample 9 series details, 1W, 2W, 3W time series 

Usefull sources:
- Overview: Wikipedia (https://en.wikipedia.org/wiki/Gradient-index_optics#cite_note-1)
- GRIN-tech Company Jena (https://www.grintech.de/wp-content/uploads/2025/05/GRIN-Lenses-Gradient-Imaging-Optics-An-Introduction-12-2022.pdf): Ressource on **basic of GRIN lenses** (with formulas for theory) made from Si- or Li-ion-exchange in glasses
- Vielleicht nützlich: https://pure.tudelft.nl/ws/portalfiles/portal/238981736/Optical_Design_of_Generalised_GRIN_Lenses_-_A_M_Boyd_-_Digital_Version.pdf
	- Doktorarbeit rund um 
- Gradient index optics: Springer Nature book ([https://doi.org/10.1007/978-3-662-04741-**5**)](https://link.springer.com/book/10.1007/978-3-662-04741-5)