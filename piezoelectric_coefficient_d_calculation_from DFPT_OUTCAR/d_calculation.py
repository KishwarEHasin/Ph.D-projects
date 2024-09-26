import numpy as np

np.set_printoptions(precision=8, suppress=True, linewidth=200)

def extract_matrix(filename, headers):
    """
    Extracts a matrix from the OUTCAR file that starts after any of the specified headers.
    The matrix continues until a line of dashes or non-numeric content is found.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Locate the line with any of the specified headers
    start_idx = None
    for idx, line in enumerate(lines):
        for header in headers:
            if header in line:
                start_idx = idx + 3  # Matrix starts 3 lines after the header (header + separator)
                break
        if start_idx is not None:
            break

    if start_idx is None:
        raise ValueError(f"None of the specified headers found in the file: {headers}")

    # Dynamically extract rows until a non-numeric line is encountered or "---" is found
    matrix = []
    for line in lines[start_idx:]:
        if '--' in line or not line.strip():  # Stop when encountering a line with '---' or an empty line
            break
        try:
            # Try to parse the line as a list of floats (skip the first column with labels like x, y, z)
            values = list(map(float, line.split()[1:]))
            matrix.append(values)
        except ValueError:
            # Stop when we can't parse a row into floats (non-numeric content)
            break

    # Convert the list to a NumPy array
    return np.array(matrix)

def calculate_inverse(C):
    """
    Calculate the elastic compliance matrix S (inverse of matrix C) if it's not singular and scale it by 1000.
    Returns the inverse matrix if it exists, else returns None.
    """
    det_C = np.linalg.det(C)

    if det_C != 0:
        inv_C = np.linalg.inv(C)
        print("Elastic compliance matrix, S_ij in 10^-3 /(GPa):\n", inv_C * 10**12) #(scaled by 10^-12)
        return inv_C  # Return the inverse matrix
    else:
        print("Matrix is singular and does not have an inverse.")
        return None  # Return None if the matrix is singular

def calculate_total_piezoelectric_tensor(e_electron, e_ion):
    """
    Calculate the total piezoelectric tensor by summing the electronic and ionic tensors.
    """
    return e_electron + e_ion

def calculate_d_alpha_i(S, e):
    """
    Compute d_{alpha i} matrix using the formula:
    d_{alpha i} = sum over j of (S_{ij} * e_{alpha j})
    """
    rows, cols = e.shape
    # Initialize a matrix to store d_{alpha i} values
    d_matrix = np.zeros((rows, S.shape[0]))  # Shape (alpha, i)

    # Loop over each row in e (corresponds to alpha in e)
    for alpha in range(rows):  # alpha = row index of e
        for i in range(S.shape[0]):  # i = row index of S
            # Sum over j for non-zero e_{alpha j}
            for j in range(cols):  # j = column index of e
                if e[alpha, j] != 0:  # Only sum for non-zero e_{alpha j}
                    d_matrix[alpha, i] += S[i, j] * e[alpha, j]

    return d_matrix

# Main code
filename = "OUTCAR-6"

with open("d_matrix", "w") as output_file:
    # Step 1: Extract the elastic moduli matrix using both possible headers
    elastic_moduli_headers = [
        " ELASTIC MODULI IONIC CONTR (kBar)", 
        " ELASTIC MODULI CONTR FROM IONIC RELAXATION (kBar)"
    ]
    
    C_raw = extract_matrix(filename, elastic_moduli_headers)
    print("Extracted Elastic Stiffness Moduli Matrix (C) in k Bar:\n", C_raw)
    output_file.write("Extracted Elastic Stiffness Moduli Matrix (C) in k Bar:\n")
    output_file.write(np.array2string(C_raw, max_line_width=200) + "\n")
    
    # Convert unit from k Bar to N/m^2 (scaling by 10^3*10^5-->10^8)
    C = C_raw * 10**8
    print("Extracted Elastic Stiffness Moduli Matrix (C) in GPa:\n", C * 10**-9) # Pa = N/m^2
    output_file.write("Extracted Elastic Stiffness Moduli Matrix (C) in GPa:\n")
    output_file.write(np.array2string(C * 10**-9, max_line_width=200) + "\n")

    # Calculate and print the S (inverse of C if it's invertible)
    S = calculate_inverse(C)  # in m^2/N 
    output_file.write("Elastic Compliance Moduli Matrix (S) in 10^-3 /(GPa) m^2/N :\n")
    if S is not None:
        output_file.write(np.array2string(S * 10**12, max_line_width=200) + "\n")
    else:
        output_file.write("Matrix is singular and does not have an inverse.\n")

    if S is not None:
        # Step 2: Extract piezoelectric tensors and calculate total
        e_electron_headers = [
            " PIEZOELECTRIC TENSOR  for field in x, y, z        (C/m^2)",
            " PIEZOELECTRIC TENSOR (including local field effects)  for field in x, y, z        (C/m^2)"
        ]
        e_ion_header = " PIEZOELECTRIC TENSOR IONIC CONTR  for field in x, y, z        (C/m^2)"
    
        # Corrected: pass e_electron_headers directly, not as a list inside another list
        e_electron = extract_matrix(filename, e_electron_headers)
        e_ion = extract_matrix(filename, [e_ion_header])

        e_tot = calculate_total_piezoelectric_tensor(e_electron, e_ion)

        print("-----------------------------------")
        print("Extracted Piezoelectric Tensor (Electron) in C/m^2:\n", e_electron)
        print("Extracted Piezoelectric Tensor (Ion) in C/m^2:\n", e_ion)
        print("Total Piezoelectric Tensor (e_tot = e_electron + e_ion) in C/m^2:\n", e_tot)
        
        output_file.write("-----------------------------------\n")
        output_file.write("Extracted Piezoelectric Tensor (Electron) in C/m^2:\n")
        output_file.write(np.array2string(e_electron, max_line_width=200) + "\n")
        output_file.write("Extracted Piezoelectric Tensor (Ion) in C/m^2:\n")
        output_file.write(np.array2string(e_ion, max_line_width=200) + "\n")
        output_file.write("Total Piezoelectric Tensor (e_tot = e_electron + e_ion) in C/m^2:\n")
        output_file.write(np.array2string(e_tot, max_line_width=200) + "\n")

        # Step 3: Calculate the d_{alpha i} matrix and change the unit 
        d_alpha_i_matrix = calculate_d_alpha_i(S, e_tot)  # in C/N

        print("-----------------------------------")
        print("Calculated d_{alpha i} matrix in pC/N:\n", d_alpha_i_matrix * 10**12)
        output_file.write("-----------------------------------\n")
        output_file.write("Calculated d_{alpha i} matrix in pC/N (or pm/V):\n")
        output_file.write(np.array2string(d_alpha_i_matrix * 10**12, max_line_width=200) + "\n")
    else:
        print("Could not calculate d_alpha_i matrix because S is None (matrix C is singular).")
        output_file.write("Could not calculate d_alpha_i matrix because S is None (matrix C is singular).\n")
