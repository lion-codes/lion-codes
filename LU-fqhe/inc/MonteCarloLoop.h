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

#ifndef MONTECARLOLOOP_H_
#define MONTECARLOLOOP_H_

#include "RunInfo.h"
#include "PermHandler.h"
#include <cuda.h>


class MonteCarloLoop {
	private:
#ifdef __DPREC__
		double *theta;
		double *phi;
		double k1,k2,r1,r2;
		double *detResult;
		static const double PI = 3.14159;
#else
		float *theta;
		float *phi;
		float k1,k2,r1,r2;
		float *detResult;
		static const float PI = 3.14159f;
#endif

		int Q,mu,nu;

		int nLoop;


	public:
		MonteCarloLoop();
		MonteCarloLoop(RunInfo rInfo, int nLoop);
		~MonteCarloLoop();

		void perform(Node *node, RunInfo *rInfo, PermHandler *pHandler, int mode);

};

#endif /* MONTECARLOLOOP_H_ */
