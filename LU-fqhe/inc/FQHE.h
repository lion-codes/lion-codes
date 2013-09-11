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


#ifndef FQHE_H_
#define FQHE_H_

#include "Node.h"
#include "RunInfo.h"
#include "MonteCarloLoop.h"
#include "PermHandler.h"

class FQHE {
	private:

		Node *node;
		RunInfo *runInfo;
		MonteCarloLoop *mcLoop;
		PermHandler *pHandler;

	public:
		FQHE();
		FQHE(int *argc,char ***argv);
		~FQHE();
		void commandLineParse(int *argc, char ***argv, int rank,int *nMC, int *nPerm, int *nMperBl, int *matSide, char **pFile, int *mode);
};



#endif /* FQHE_H_ */
