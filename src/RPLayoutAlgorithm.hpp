/*
 ==============================================================================

 RPLayoutAlgorithm.hpp
 Copyright (C) 2016, 2017  G. Brinkmann

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

#ifndef RPLayoutAlgorithm_hpp
#define RPLayoutAlgorithm_hpp

#include "RPGraphLayout.hpp"

namespace RPGraph
{
    class LayoutAlgorithm
    {
    public:
        LayoutAlgorithm(GraphLayout &layout);
        ~LayoutAlgorithm();
        GraphLayout &layout;
        
        virtual void sync_layout() = 0; // write current layout to `layout'.
    };
}

#endif
