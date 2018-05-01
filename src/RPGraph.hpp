/*
 ==============================================================================

 RPGraph.hpp
 Copyright © 2016, 2017, 2018  G. Brinkmann

 This file is part of graph_viewer.

 graph_viewer is free software: you can redistribute it and/or modify
 it under the terms of version 3 of the GNU Affero General Public License as
 published by the Free Software Foundation.

 graph_viewer is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with graph_viewer.  If not, see <https://www.gnu.org/licenses/>.

 ==============================================================================
*/


#ifndef RPGraph_hpp
#define RPGraph_hpp
#include <vector>
#include <string>
#include <unordered_map>

namespace RPGraph
{
    // Type to represent node IDs.
    // NOTE: we limit to 4,294,967,296 nodes through uint32_t.
    typedef uint32_t nid_t;

    // Type to represent edge IDs.
    // NOTE: uint32_t limits density to 50% for directed graphs.
    typedef uint32_t eid_t;

    // Virtual base class to derive different Graph types from.
    class Graph
    {
        public:
            virtual nid_t num_nodes() = 0;
            virtual nid_t num_edges() = 0;
            virtual nid_t degree(nid_t nid) = 0;
            virtual nid_t in_degree(nid_t nid) = 0;
            virtual nid_t out_degree(nid_t nid) = 0;
            virtual std::vector<nid_t> neighbors_with_geq_id(nid_t nid) = 0;

    };

    // Very basic (adjacency list) representation of an undirected graph.
    class UGraph : public Graph
    {
    private:
        nid_t node_count, edge_count;
        std::unordered_map<nid_t, nid_t> degrees;
        std::unordered_map<nid_t, std::vector<nid_t>> adjacency_list;

        bool has_node(nid_t nid);
        bool has_edge(nid_t s, nid_t t);
        void add_node(nid_t nid);
        void add_edge(nid_t s, nid_t t);

    public:
        // Construct UGraph from edgelist. IDs in edgelist are mapped to
        // [0, 1, ..., num_nodes-1]. Removes any self-edges.
        UGraph(std::string edgelist_path);
        std::unordered_map<nid_t, nid_t> node_map; // el id -> UGraph id
        std::unordered_map<nid_t, nid_t> node_map_r; // UGraph id -> el id

        virtual nid_t num_nodes() override;
        virtual nid_t num_edges() override;
        virtual nid_t degree(nid_t nid) override;
        virtual nid_t in_degree(nid_t nid) override;
        virtual nid_t out_degree(nid_t nid) override;

        std::vector<nid_t> neighbors_with_geq_id(nid_t nid) override;
    };

    // Compressed sparserow (CSR) for undirected graphs.
    class CSRUGraph : public Graph
    {
    private:
        nid_t *edges;   // All edgelists, concatenated.
        nid_t *offsets; // For each node, into edges.
        nid_t node_count, edge_count;
        nid_t first_free_id, edges_seen;

    public:
        std::unordered_map<nid_t, nid_t> nid_to_offset;
        nid_t *offset_to_nid;

        CSRUGraph(nid_t num_nodes, nid_t num_edges);
        ~CSRUGraph();

        /// Inserts node_id and its edges. Once inserted, edges
        /// can't be altered for this node.
        void insert_node(nid_t node_id, std::vector<nid_t> nbr_ids);
        void fix_edge_ids(); // this should go...

        virtual nid_t num_nodes() override;
        virtual nid_t num_edges() override;
        virtual nid_t degree(nid_t nid) override;
        virtual nid_t in_degree(nid_t nid) override;
        virtual nid_t out_degree(nid_t nid) override;

        nid_t nbr_id_for_node(nid_t nid, nid_t nbr_no);
    };
}

#endif /* Graph_h */
