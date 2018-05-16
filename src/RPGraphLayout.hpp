/*
 ==============================================================================

 RPGraphLayout.cpp
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

#ifndef RPGraphLayout_hpp
#define RPGraphLayout_hpp

#include "RPGraph.hpp"
#include "RPCommon.hpp"
#include <string>

namespace RPGraph
{
    class GraphLayout
    {
    private:
        Coordinate *coordinates;

    protected:
        float width, height;
        float minX(), minY(), maxX(), maxY();

    public:
        GraphLayout(RPGraph::UGraph &graph, float width, float height);
        ~GraphLayout();

        UGraph &graph; // to lay-out

        // randomize the layout position of all nodes.
        void randomizePositions();

        float getX(nid_t node_id), getY(nid_t node_id);
        float getXRange(), getYRange(), getSpan();
        float getDistance(nid_t n1, nid_t n2);
        Real2DVector getDistanceVector(nid_t n1, nid_t n2);
        Real2DVector getNormalizedDistanceVector(nid_t n1, nid_t n2);
        Coordinate getCoordinate(nid_t node_id);
        Coordinate getCenter();


        void setX(nid_t node_id, float x_value), setY(nid_t node_id, float y_value);
        void moveNode(nid_t, Real2DVector v);
        void setCoordinates(nid_t node_id, Coordinate c);
        void writeToPNG(const int image_w, const int image_h, std::string path);
        void writeToCSV(std::string path);
        void writeToBin(std::string path);
    };
}

#endif /* RPGraphLayout_hpp */
