#Python program to calculate solvent accessible surface area of 1ASY from three files - 1ASY_C.asa, 1ASY_P.asa, 1ASY_R.asa

fc=open('1ASY_C.asa','r')									#
clines=fc.readlines()										#
fc.close()											#
												#
fp=open('1ASY_P.asa','r')									#	Reading the
plines=fp.readlines()										#	.asa files
fp.close()											#
												#
fr=open('1ASY_R.asa','r')										#
rlines=fr.readlines()										#
fr.close()											#

cplines=[]											#
crlines=[]											#	Dividing the lines
for i in clines:										#	from the .asa file
	if len(i[17:20].strip())==3:								#	of the complex into
		cplines.append(i)								#	protein and rna parts
	else:											#
		crlines.append(i)								#

if len(plines)==len(cplines) and len(rlines)==len(crlines):					#	[OPTIONAL]
	pass											#	Checking for
else:												#	consistency
	raise ValueError('Inconsistent data in the input files')				#


#--------Computations for the interface area of the protein part-------------------------------------------------------------------------------------------------

p_int=[]
p_int_atoms=0
p_int_res=[]
p_area=0
for i in range(len(plines)):
	if int(plines[i][6:11].strip())==int(cplines[i][6:11].strip()) and plines[i][11:17].strip()==cplines[i][11:17].strip() and plines[i][21]==cplines[i][21]:
		da=round(float(plines[i][54:62].strip())-float(cplines[i][54:62].strip()),3)
		if da>0:
			p_int.append(plines[i][:62]+cplines[i][54:62]+str(da).rjust(8,' '))
			p_int_atoms+=1
			p_area+=da
			temp_residue=plines[i][21:28].strip()
			if temp_residue not in p_int_res:
				p_int_res.append(temp_residue)
p_int_resnum=len(p_int_res)

#--------Creation of the .int file for the protein part----------------------------------------------------------------------------------------------------------

fp_int=open(protein_file[:-4]+'.int','w')
for each in p_int:
	fp_int.write(each+'\n')
fp_int.close()


#--------Computations for the interface area of the RNA part-----------------------------------------------------------------------------------------------------

r_int=[]
r_int_atoms=0
r_int_nus=[]
r_area=0
for i in range(len(rlines)):
	if int(rlines[i][6:11].strip())==int(crlines[i][6:11].strip()) and rlines[i][11:17].strip()==crlines[i][11:17].strip() and rlines[i][21]==crlines[i][21]:
		da=round(float(rlines[i][54:62].strip())-float(crlines[i][54:62].strip()),3)
		if da>0:
			r_int.append(rlines[i][:62]+crlines[i][54:62]+str(da).rjust(8,' '))
			r_int_atoms+=1
			r_area+=da
			temp_nucleotide=rlines[i][21:28].strip()
			if temp_nucleotide not in r_int_nus:
				r_int_nus.append(temp_nucleotide)
r_int_nusnum=len(r_int_nus)

#--------Creation of the .int file for the RNA part--------------------------------------------------------------------------------------------------------------

fr_int=open(rna_file[:-4]+'.int','w')
for each in r_int:
	fr_int.write(each+'\n')
fr_int.close()
