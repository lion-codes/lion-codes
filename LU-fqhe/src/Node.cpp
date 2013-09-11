/*
 *  Copyright 2013 William J. Brouwer, Pierre-Yves Taunay
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <stdexcept>
#include <iostream>

#include "Node.h"
#include "LocalInfo.h"
#include "GlobalInfo.h"

using namespace std;

Node::Node() {
	try {
		lInfo = new LocalInfo();
		gInfo = new GlobalInfo();
	}
	catch(bad_alloc& e) {
		cerr << "ERROR \t Failed to allocate LocalInfo() or GlobalInfo() in Node() in Node.cpp" << endl;
		throw exception();
	}
}

GlobalInfo* Node::getGlobalInfo() {
	return gInfo;
}

LocalInfo* Node::getLocalInfo() {
	return lInfo;
}

Node::~Node() {
	if(lInfo != NULL) {
		delete lInfo;
		lInfo = NULL;
	}

	if(gInfo != NULL) {
		delete gInfo;
		gInfo = NULL;
	}
}
