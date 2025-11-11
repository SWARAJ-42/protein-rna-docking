# --------------------------------------------------------------------------------------------------
# Python Program to Calculate Solvent Accessible Surface Area (SASA)
# for the complex 1ASY from three files:
#   - 1ASY_C.asa : Complex
#   - 1ASY_P.asa : Protein
#   - 1ASY_R.asa : RNA
# --------------------------------------------------------------------------------------------------

# ---- Reading the .asa files ----------------------------------------------------------------------

with open('./1ASY/1ASY_C.asa', 'r') as fc:
    clines = fc.readlines()

with open('./1ASY/1ASY_P.asa', 'r') as fp:
    plines = fp.readlines()

with open('./1ASY/1ASY_R.asa', 'r') as fr:
    rlines = fr.readlines()

# ---- Dividing complex .asa data into protein and RNA parts ---------------------------------------

cplines = []
crlines = []

for line in clines:
    # If residue name has 3 letters, it's a protein; otherwise, RNA
    if len(line[17:20].strip()) == 3:
        cplines.append(line)
    else:
        crlines.append(line)

# ---- Consistency check between datasets ----------------------------------------------------------

if len(plines) != len(cplines) or len(rlines) != len(crlines):
    raise ValueError('Inconsistent data in the input files')

# --------------------------------------------------------------------------------------------------
# -------- Computations for the Interface Area of the Protein Part ---------------------------------
# --------------------------------------------------------------------------------------------------

p_int = []
p_int_atoms = 0
p_int_res = []
p_area = 0.0

for i in range(len(plines)):
    if (
        int(plines[i][6:11].strip()) == int(cplines[i][6:11].strip()) and
        plines[i][11:17].strip() == cplines[i][11:17].strip() and
        plines[i][21] == cplines[i][21]
    ):
        delta_area = round(float(plines[i][54:62].strip()) - float(cplines[i][54:62].strip()), 3)
        if delta_area > 0:
            p_int.append(plines[i][:62] + cplines[i][54:62] + str(delta_area).rjust(8, ' '))
            p_int_atoms += 1
            p_area += delta_area

            residue = plines[i][21:28].strip()
            if residue not in p_int_res:
                p_int_res.append(residue)

p_int_resnum = len(p_int_res)

# ---- Writing interface data for protein part -----------------------------------------------------

protein_file = '1ASY_P.asa'
with open(protein_file[:-4] + '.int', 'w') as fp_int:
    for record in p_int:
        fp_int.write(record + '\n')

# --------------------------------------------------------------------------------------------------
# -------- Computations for the Interface Area of the RNA Part -------------------------------------
# --------------------------------------------------------------------------------------------------

r_int = []
r_int_atoms = 0
r_int_nus = []
r_area = 0.0

for i in range(len(rlines)):
    if (
        int(rlines[i][6:11].strip()) == int(crlines[i][6:11].strip()) and
        rlines[i][11:17].strip() == crlines[i][11:17].strip() and
        rlines[i][21] == crlines[i][21]
    ):
        delta_area = round(float(rlines[i][54:62].strip()) - float(crlines[i][54:62].strip()), 3)
        if delta_area > 0:
            r_int.append(rlines[i][:62] + crlines[i][54:62] + str(delta_area).rjust(8, ' '))
            r_int_atoms += 1
            r_area += delta_area

            nucleotide = rlines[i][21:28].strip()
            if nucleotide not in r_int_nus:
                r_int_nus.append(nucleotide)

r_int_nusnum = len(r_int_nus)

# ---- Writing interface data for RNA part ---------------------------------------------------------

rna_file = '1ASY_R.asa'
with open(rna_file[:-4] + '.int', 'w') as fr_int:
    for record in r_int:
        fr_int.write(record + '\n')

# --------------------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------------------
print(f"Protein Interface: {p_int_atoms} atoms, {p_int_resnum} residues, Total Area = {p_area:.3f}")
print(f"RNA Interface:     {r_int_atoms} atoms, {r_int_nusnum} nucleotides, Total Area = {r_area:.3f}")