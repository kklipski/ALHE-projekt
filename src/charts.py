import plotly.graph_objs as go
import plotly.offline as py

"""
Funkcja create_diagram tworzy diagram zakladajac,
ze w parametrze zostaly przekazane dwie listy czasow
dla dwoch roznych algorytmow oraz lista
wykorzystanej liczby wierzcholkow w kolejnych iteracjach
"""


def create_times_diagram(scores, episodes,exploit, exploit_x, explore, explore_x, standing, standing_x, output):
    trace = []

    for idx, s in enumerate(scores):
        trace.append( go.Scatter(
            y = s,
            x = episodes,
            name="Sieć nr " + str(idx)
        ))
    for idx, e in enumerate(exploit):
        trace.append(go.Scatter(
            y = e,
            x = exploit_x[idx],
            name="Exploit sieci " + str(idx),
            mode = 'markers',
            marker={'symbol':1, 'size':10}
        ))
    for idx, e in enumerate(explore):
        trace.append(go.Scatter(
            y = e,
            x = explore_x[idx],
            name="Explore sieci " + str(idx),
            mode = 'markers',
            marker={'symbol':200, 'size':20}
        ))

    data = trace

    layout = go.Layout(
        barmode='group'
    )

    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename=output)


def parser(file_name, n_networks):
    f = open(file_name, 'r')
    scores = [[] for _ in range(n_networks)]
    exploit = [[] for _ in range(n_networks)]
    exploit_x = [[] for _ in range(n_networks)]
    explore = [[] for _ in range(n_networks)]
    explore_x = [[] for _ in range(n_networks)]
    standing = [[] for _ in range(n_networks)]
    standing_x = [[] for _ in range(n_networks)]

    for line in f:
        splitted = line.split(',')

        net = int(splitted[0])
        if len(splitted) > 2:
            scores[net].append(splitted[2])
        if line.find("exploit", 0 , len(line) - 1) != -1:
            exploit[net].append(splitted[2])  # dodaj score
            exploit_x[net].append(len(scores[net])-1) # dla takiego epizodu
        if line.find("explore", 0 , len(line) - 1) != -1:
            explore[net].append(splitted[2])  # dodaj score
            explore_x[net].append(len(scores[net])-1) # dla takiego epizodu

    episodes = len(scores[0])

    return scores, episodes, exploit, exploit_x, explore, explore_x, standing, standing_x


def main():
    scores, episodes, exploit, exploit_x, explore, explore_x, standing, standing_x = parser("log.txt", 5)
    create_times_diagram(scores, [i for i in range(episodes)],exploit, exploit_x, explore, explore_x, standing, 
                         standing_x, "chart.html")


if __name__ == "__main__":
    main()
