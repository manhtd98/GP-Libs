import pygraphviz as pgv
node = {}
count_operator = {'+':0, '-':0, "*":0, '/': 0}
with open("dag.txt") as f:
    
    for lines in f.readlines():
        line = lines.split()
        centor = str(count_operator[line[3]]) + line[3]
        node[line[0]] = {centor:None}
        node[centor] = {line[2]:None,line[4]:None}
        # node[centor] = { }
        # node[centor] = { : None }
        
        count_operator[line[3]]+=1
        
G = pgv.AGraph(node, strict=False, directed=False)
G.layout(prog="dot")
G.draw(f"tree-dag.png")