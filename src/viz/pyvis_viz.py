
# ##https://gist.github.com/maciejkos/e3bc958aac9e7a245dddff8d86058e17
# def draw_graph3(networkx_graph,notebook=True,output_filename='graph.html',show_buttons=False,only_physics_buttons=False):
#         """
#         This function accepts a networkx graph object,
#         converts it to a pyvis network object preserving its node and edge attributes,
#         and both returns and saves a dynamic network visualization.

#         Valid node attributes include:
#             "size", "value", "title", "x", "y", "label", "color".

#             (For more info: https://pyvis.readthedocs.io/en/latest/documentation.html#pyvis.network.Network.add_node)

#         Valid edge attributes include:
#             "arrowStrikethrough", "hidden", "physics", "title", "value", "width"

#             (For more info: https://pyvis.readthedocs.io/en/latest/documentation.html#pyvis.network.Network.add_edge)


#         Args:
#             networkx_graph: The graph to convert and display
#             notebook: Display in Jupyter?
#             output_filename: Where to save the converted network
#             show_buttons: Show buttons in saved version of network?
#             only_physics_buttons: Show only buttons controlling physics of network?
#         """

#         # import
#         from pyvis import network as net

#         # make a pyvis network
#         pyvis_graph = net.Network(notebook=notebook)
#         pyvis_graph.width = '1000px'
#         # for each node and its attributes in the networkx graph
#         for node,node_attrs in networkx_graph.nodes(data=True):
#             pyvis_graph.add_node(node,**node_attrs)
#     #         print(node,node_attrs)

#         # for each edge and its attributes in the networkx graph
#         for source,target,edge_attrs in networkx_graph.edges(data=True):
#             # if value/width not specified directly, and weight is specified, set 'value' to 'weight'
#             if not 'value' in edge_attrs and not 'width' in edge_attrs and 'weight' in edge_attrs:
#                 # place at key 'value' the weight of the edge
#                 edge_attrs['value']=edge_attrs['weight']
#             # add the edge
#             pyvis_graph.add_edge(source,target,**edge_attrs)

#         # turn buttons on
#         if show_buttons:
#             if only_physics_buttons:
#                 pyvis_graph.show_buttons(filter_=['physics'])
#             else:
#                 pyvis_graph.show_buttons()

#         # return and also save
#         return pyvis_graph.show(output_filename)


def draw_graph3(networkx_graph,notebook=True,output_filename='graph.html',show_buttons=True,only_physics_buttons=False,
                height=None,width=None,bgcolor=None,font_color=None,pyvis_options=None):
    """
    This function accepts a networkx graph object,
    converts it to a pyvis network object preserving its node and edge attributes,
    and both returns and saves a dynamic network visualization.
    Valid node attributes include:
        "size", "value", "title", "x", "y", "label", "color".
        (For more info: https://pyvis.readthedocs.io/en/latest/documentation.html#pyvis.network.Network.add_node)
    Valid edge attributes include:
        "arrowStrikethrough", "hidden", "physics", "title", "value", "width"
        (For more info: https://pyvis.readthedocs.io/en/latest/documentation.html#pyvis.network.Network.add_edge)
    Args:
        networkx_graph: The graph to convert and display
        notebook: Display in Jupyter?
        output_filename: Where to save the converted network
        show_buttons: Show buttons in saved version of network?
        only_physics_buttons: Show only buttons controlling physics of network?
        height: height in px or %, e.g, "750px" or "100%
        width: width in px or %, e.g, "750px" or "100%
        bgcolor: background color, e.g., "black" or "#222222"
        font_color: font color,  e.g., "black" or "#222222"
        pyvis_options: provide pyvis-specific options (https://pyvis.readthedocs.io/en/latest/documentation.html#pyvis.options.Options.set)
    """

    # import
    from pyvis import network as net

    # make a pyvis network
    network_class_parameters = {"notebook": notebook, "height": height, "width": width, "bgcolor": bgcolor, "font_color": font_color}
    pyvis_graph = net.Network(**{parameter_name: parameter_value for parameter_name, parameter_value in network_class_parameters.items() if parameter_value})

    # for each node and its attributes in the networkx graph
    for node,node_attrs in networkx_graph.nodes(data=True):
        pyvis_graph.add_node(node,**node_attrs)

    # for each edge and its attributes in the networkx graph
    for source,target,edge_attrs in networkx_graph.edges(data=True):
        # if value/width not specified directly, and weight is specified, set 'value' to 'weight'
        if not 'value' in edge_attrs and not 'width' in edge_attrs and 'weight' in edge_attrs:
            # place at key 'value' the weight of the edge
            edge_attrs['value']=edge_attrs['weight']
        # add the edge
        pyvis_graph.add_edge(source,target,**edge_attrs)

    # turn buttons on
    if show_buttons:
        if only_physics_buttons:
            pyvis_graph.show_buttons(filter_=['physics'])
        else:
            pyvis_graph.show_buttons()

    # pyvis-specific options
    if pyvis_options:
        pyvis_graph.set_options(pyvis_options)

    # return and also save
    return pyvis_graph.show(output_filename)