import csv

# Входной и выходной файлы
input_file = "C:/Proj_edge_distr/deezer_europe/deezer_europe_edges.csv"
output_file = "C:/Proj_edge_distr/deezer_europe_edges.edges"

# Читаем CSV и записываем в .edge
with open(input_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", encoding="utf-8") as f_out:
    
    reader = csv.reader(f_in)
    
    # Пропускаем заголовок (node_1,node_2)
    header = next(reader, None)
    
    # Записываем каждое ребро в формате "узел1 узел2"
    for row in reader:
        if len(row) >= 2:
            node1 = row[0].strip()
            node2 = row[1].strip()
            f_out.write(f"{node1} {node2}\n")

print(f"Готово! Файл сохранён как {output_file}")