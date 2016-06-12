from csv import reader, writer


def get_dict_of_dicts_from_csv_matrix(file_name, delimiter=','):
    with open(file_name, 'rU') as fh:
        all_rows = reader(fh, delimiter=delimiter, quotechar='"')
        cols = all_rows.next()[1:]
        rem_rows = [x for x in all_rows]
    return {l[0]: {cols[i]: elem for i, elem in enumerate(l[1:])} for l in rem_rows}


def csv_matrix_from_dict_of_dicts(matrix, file_name, delimiter=','):
    rows = matrix.keys()
    cols = matrix[rows[0]].keys()
    with open(file_name, "w") as fh:
        wr = writer(fh, delimiter=delimiter, quotechar='"', dialect='excel')
        wr.writerow([''] + cols)
        for row in rows:
            wr.writerow([row] + [matrix[row][col] for col in cols])
    return 0


f1_name = "/Users/cover_drive/Downloads/TitlesGraph(Original).csv - Final with Ajit's Correction.csv"
f2_name = "/Users/cover_drive/Downloads/TitlesGraph(Original).csv - Madhav's edits.csv"
d1 = get_dict_of_dicts_from_csv_matrix(f1_name)
d2 = get_dict_of_dicts_from_csv_matrix(f2_name)
out = [('Profile', 'Job', 'Old', 'New')] + \
    [(k1, k2, float(val), float(d2[k1][k2])) for k1, in_d in d1.iteritems() for k2, val in in_d.iteritems() if float(val) != float(d2[k1][k2])]
out_name = "/Users/cover_drive/temp.csv"
with open(out_name, "w") as fh:
    wr = writer(fh, delimiter=',', quotechar='"', dialect='excel')
    for row in out:
        wr.writerow(row)

# out_name = "/Users/cover_drive/temp.csv"
# csv_matrix_from_dict_of_dicts(d, out_name)
