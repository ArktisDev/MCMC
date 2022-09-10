import copy
import os
import shutil

class PDF:
    __global_id_counter = 0
    
    @classmethod
    def __NewUniqueID(cls) -> int:
        res = PDF.__global_id_counter
        PDF.__global_id_counter += 1
        return res
        
    def __init__(self, funcParams, funcName) -> None:
        self.uniqueIdentifier = PDF.__NewUniqueID()
        
        self.funcParams = funcParams
        self.funcName = funcName
        
    def ModifyParameterValue(self, name : str, newValue) -> "PDF":
        selfcopy = self.Copy()
        found = False
        for parameter in selfcopy.funcParams:
            if parameter['name'] == name:
                parameter['value'] = newValue
                found = True
        if not found:
            raise Exception('Could not find paramter ' + name + '')
        return selfcopy
    
    def GenerateCode(self) -> str:
        res = '__device__ __host__ float ' + self.GetFunctionName() + '(float r) {\n'
        for variable in self.funcParams:
            res += '\t' + variable['type'] + ' ' + variable['name'] + ' = '
            if (variable['type'] == 'int'):
                res +=  str(int(variable['value'])) + ';\n'
            elif(variable['type'] == 'float'):
                res +=  str(float(variable['value'])) + ';\n'
            else:
                raise Exception('Type ' + variable['type'] + ' is unknown')
        
        res += '\treturn ' + self.funcName + '(r'
        
        for variable in self.funcParams:
            res += ', ' + variable['name']
            
        res += ');\n}\n'
        
        return res
        
    def GetFunctionName(self) -> str:
        return 'PDF_' + self.funcName + '_' + str(self.uniqueIdentifier)
    
    def GetNewID(self) -> None:
        self.uniqueIdentifier = PDF.__NewUniqueID()
        pass
    
    def Copy(self) -> "PDF":
        newObject = copy.deepcopy(self)
        newObject.uniqueIdentifier = PDF.__NewUniqueID()
        return newObject

element_list = ['H', 'HE', 'LI', 'BE', 'B', 'C', 'N', 'O', 'F', 'NE', 'NA',
                'MG', 'AL', 'SI', 'P', 'S', 'CL', 'AR', 'K', 'CA', 'SC', 'TI',
                'V', 'CR', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN', 'GA', 'GE',
                'AS', 'SE', 'BR', 'KR', 'RB', 'SR', 'Y', 'ZR', 'NB', 'MO',
                'TC', 'RU', 'RH', 'PD', 'AG', 'CD', 'IN', 'SN', 'SB', 'TE',
                'I', 'XE', 'CS', 'BA', 'LA', 'CE', 'PR', 'ND', 'PM', 'SM',
                'EU', 'GD', 'TB', 'DY', 'HO', 'ER', 'TM', 'YB', 'LU', 'HF',
                'TA', 'W', 'RE', 'OS', 'IR', 'PT', 'AU', 'HG', 'TL', 'PB',
                'BI', 'PO', 'AT', 'RN', 'FR', 'RA', 'AC', 'TH', 'PA', 'U',
                'NP', 'PU', 'AM', 'CM', 'BK', 'CT', 'ES', 'FM', 'MD', 'NO',
                'LR', 'RF', 'DB', 'SG', 'BH', 'HS', 'MT', 'DS', 'RG', 'CN',
                'NH', 'FL', 'MC', 'LV', 'TS', 'OG']
atomic_number_dict = {k: v + 1 for v, k in enumerate(element_list)}
atomic_symbol_dict = {v: k for k, v in atomic_number_dict.items()}

def NuclearZAndA(element) -> dict:
    element = element.split('-')
    symbol = element[1] if element[0].isdigit() else element[0]
    A = int(element[0]) if element[0].isdigit() else int(element[1])
    Z = atomic_number_dict[symbol.upper()]
    return {'Z':Z, 'A':A}

class IntegrationSettings:
    def __init__(self) -> None:
        # Init everything to None, then the user will change whatever.
        self.nNeutronsA = None
        self.nProtonsA = None
        self.nNeutronsB = None
        self.nProtonsB = None
        
        self.NeutronPDFListA = None
        self.ProtonPDFListA = None
        self.NeutronPDFListB = None
        self.ProtonPDFListB = None
        
        self.b0 = None
        self.db = None
        self.ndbs = None
        
        self.overrideSeed = False
        self.newSeed = 314159265
        
        self.totalBlocks = None
        self.threadsPerBlock = None
        self.samplesPerThread = None
        self.totalRuns = None
        self.samplesForMix = None
        
        # Usage:
        # outDir = '/home/user/'
        # outFile = 'file.dat'
        self.outDir = None
        self.outFile = None
        
        # For where to put the actual generated macro
        self.macroOutDir = './'
        self.macroName = 'macro0.mac'
        
        self.IntegrateRMS = None
        self.IntegrateSAB = None
        self.IntegrateIAB = None
        self.IntegrateJAB = None
        pass
    
    def __CreatePDFListA(self) -> None:
        if self.nNeutronsA is None:
            raise Exception('Number of neutrons in nucleus A is not set')
        if self.nProtonsA is None:
            raise Exception('Number of protons in nucleus A is not set')
        self.NeutronPDFListA = [None] * self.nNeutronsA
        self.ProtonPDFListA = [None] * self.nProtonsA
        pass
    
    def __CreatePDFListB(self) -> None:
        if self.nNeutronsB is None:
            raise Exception('Number of neutrons in nucleus A is not set')
        if self.nProtonsB is None:
            raise Exception('Number of protons in nucleus A is not set')
        self.NeutronPDFListB = [None] * self.nNeutronsB
        self.ProtonPDFListB = [None] * self.nProtonsB
        pass
    
    def SetIsotopeA(self, isotope) -> None:
        zna = NuclearZAndA(isotope)
        self.nNeutronsA = zna['A'] - zna['Z']
        self.nProtonsA = zna['Z']
        self.__CreatePDFListA()
        pass
        
    def SetIsotopeB(self, isotope) -> None:
        zna = NuclearZAndA(isotope)
        self.nNeutronsB = zna['A'] - zna['Z']
        self.nProtonsB = zna['Z']
        self.__CreatePDFListB()
        pass
        
    def SetPDFA(self, pdf : PDF) -> None:
        self.NeutronPDFListA = [pdf] * self.nNeutronsA
        self.ProtonPDFListA = [pdf] * self.nProtonsA
        pass
    
    def SetNeutronPDFA(self, pdf : PDF) -> None:
        self.NeutronPDFListA = [pdf] * self.nNeutronsA
        pass
    
    def SetProtonPDFA(self, pdf : PDF) -> None:
        self.ProtonPDFListA = [pdf] * self.nProtonsA
        pass
    
    def SetNeutronPDFA(self, pdf : PDF, neutron : int) -> None:
        self.NeutronPDFListA[neutron] = pdf
        pass
    
    def SetProtonPDFA(self, pdf : PDF, proton : int) -> None:
        self.ProtonPDFListA[proton] = pdf
        pass
    
    def SetPDFB(self, pdf : PDF) -> None:
        self.NeutronPDFListB = [pdf] * self.nNeutronsB
        self.ProtonPDFListB = [pdf] * self.nProtonsB
        pass
    
    def SetNeutronPDFB(self, pdf : PDF) -> None:
        self.NeutronPDFListB = [pdf] * self.nNeutronsB
        pass
    
    def SetProtonPDFB(self, pdf : PDF) -> None:
        self.ProtonPDFListB = [pdf] * self.nProtonsB
        pass
    
    def SetNeutronPDFB(self, pdf : PDF, neutron : int) -> None:
        self.NeutronPDFListB[neutron] = pdf
        pass
    
    def SetProtonPDFB(self, pdf : PDF, proton : int) -> None:
        self.ProtonPDFListB[proton] = pdf
        pass
    
    def SetAllPDFs(self, pdf : PDF) -> None:
        self.SetPDFA(pdf)
        self.SetPDFB(pdf)
        pass
    
    def SetImpactParamterStepping(self, b0 : float, db : float, ndbs : int) -> None:
        self.b0 = b0
        self.db = db
        self.ndbs = ndbs
        pass
        
    def SetOutputDir(self, dir : str) -> None:
        self.outDir = dir
        pass
    
    def SetOutputFile(self, file : str) -> None:
        self.outFile = file
        pass
    
    def SetMacroOutputDir(self, dir : str) -> None:
        self.macroOutDir = dir
        pass
    
    def SetMacroName(self, file : str) -> None:
        self.macroName = file
        pass
    
    def SetIntegrationTypes(self, RMS : bool, SAB : bool, IAB : bool, JAB : bool) -> None:
        self.IntegrateRMS = RMS
        self.IntegrateSAB = SAB
        self.IntegrateIAB = IAB
        self.IntegrateJAB = JAB
        pass
    
    def SetMarkovMixLength(self, mixLength : int) -> None:
        self.samplesForMix = mixLength
        pass
    
    def SetGPUKernelParams(self, totalRuns : int, totalBlocks : int, 
                           threadsPerBlock : int, samplesPerThread : int):
        self.totalRuns = totalRuns
        self.totalBlocks = totalBlocks
        self.threadsPerBlock = threadsPerBlock
        self.samplesPerThread = samplesPerThread
        pass
    
    def IsValid(self) -> bool:
        valid = True
        for key, value in self.__dict__.items():
            if value is None:
                print('Attribute ' + key + ' is not set yet')
                valid = False
        if self.NeutronPDFListA is not None:
            if any(elem is None for elem in self.NeutronPDFListA):
                print('NeutronPDFListA contains a None PDF')
                valid = False
        if self.NeutronPDFListB is not None:
            if any(elem is None for elem in self.NeutronPDFListB):
                print('NeutronPDFListB contains a None PDF')
                valid = False
        if self.ProtonPDFListA is not None:
            if any(elem is None for elem in self.ProtonPDFListA):
                print('ProtonPDFListA contains a None PDF')
                valid = False
        if self.ProtonPDFListB is not None:
            if any(elem is None for elem in self.ProtonPDFListB):
                print('ProtonPDFListB contains a None PDF')
                valid = False
        return valid
    
    def Copy(self):
        res = copy.deepcopy(self)
        # But the PDF's didn't get new identifiers correctly
        uniquePDFs = set(res.ProtonPDFListA + res.ProtonPDFListB + res.NeutronPDFListA + res.NeutronPDFListB)
        uniquePDFs = list(uniquePDFs)
        
        for pdf in uniquePDFs:
            pdf.GetNewID()
        return res
        
    def PrintIsotopes(self):
        print('Nucleus A is ' + atomic_symbol_dict[self.nProtonsA] + '-' + str(self.nProtonsA + self.nNeutronsA))
        print('Nucleus B is ' + atomic_symbol_dict[self.nProtonsB] + '-' + str(self.nProtonsB + self.nNeutronsB))
        
    def GenerateCode(self) -> str:
        constint = 'const int '
        constfloat = 'const float '
        string = 'std::string '
        constbool = 'const bool '
        constexprauto = 'constexpr static auto '
        tbConstArr = 'TemplateBuilder::ConstArray'
        tbConcat = 'TemplateBuilder::ConcatArray'
        
        res = '#ifdef EXTRACT_MACRO_FUNCTIONS\n'
        
        # The unique PDF's which show up in this collision
        uniquePDFs = set(self.ProtonPDFListA + self.ProtonPDFListB + self.NeutronPDFListA + self.NeutronPDFListB)
        uniquePDFs = list(uniquePDFs)
        
        for pdf in uniquePDFs:
            res += pdf.GenerateCode() + '\n'

        res += "#else\n"

        res += constint + 'nNeutronA = ' + str(self.nNeutronsA) + ';\n'
        res += constint + 'nProtonA = ' + str(self.nProtonsA) + ';\n'
        res += constint + 'nNeutronB = ' + str(self.nNeutronsB) + ';\n'
        res += constint + 'nProtonB = ' + str(self.nProtonsB) + ';\n\n'
        
        res += constint + 'nNucleonsA = nNeutronA + nProtonA;\n'
        res += constint + 'nNucleonsB = nNeutronB + nProtonB;\n'
        res += constint + 'nNucleons = nNucleonsA + nNucleonsB;\n'
        
        # Specify each of the pdf's, keep track of which ones I generate
        fullPDFList = []
        i = 0
        for pdf in self.ProtonPDFListA:
            res += constexprauto + 'pdfArrAP' + str(i) + ' = ' + tbConstArr + '<1>(' + pdf.GetFunctionName() + ');\n'
            fullPDFList.append('pdfArrAP' + str(i))
            i += 1
        i = 0
        for pdf in self.NeutronPDFListA:
            res += constexprauto + 'pdfArrAN' + str(i) + ' = ' + tbConstArr + '<1>(' + pdf.GetFunctionName() + ');\n'
            fullPDFList.append('pdfArrAN' + str(i))
            i += 1
        i = 0
        for pdf in self.ProtonPDFListB:
            res += constexprauto + 'pdfArrBP' + str(i) + ' = ' + tbConstArr + '<1>(' + pdf.GetFunctionName() + ');\n'
            fullPDFList.append('pdfArrBP' + str(i))
            i += 1
        i = 0
        for pdf in self.NeutronPDFListB:
            res += constexprauto + 'pdfArrBN' + str(i) + ' = ' + tbConstArr + '<1>(' + pdf.GetFunctionName() + ');\n'
            fullPDFList.append('pdfArrBN' + str(i))
            i += 1
            
        # Now concat them into the final list of pdf's
        i = 0
        res += '\n' + constexprauto + 'pdfArrC' + str(i) + ' = ' + tbConcat + '(' + fullPDFList[0] + ', ' + fullPDFList[1] + ');\n'
        i += 1
        for pdf in fullPDFList[2:]:
            res += constexprauto  + 'pdfArrC' + str(i) + ' = ' + tbConcat + '(pdfArrC' + str(i - 1) + ', ' + pdf + ');\n'
            i += 1
        
        res += constexprauto + 'pdfArr = pdfArrC' + str(i - 1) + ';\n\n'
        
        res += constint + 'totalBlocks = ' + str(self.totalBlocks) + ';\n'
        res += constint + 'threadsPerBlock = ' + str(self.threadsPerBlock) + ';\n'
        res += constint + 'samplesPerThread = ' + str(self.samplesPerThread) + ';\n'
        res += constint + 'totalRuns = ' + str(self.totalRuns) + ';\n'
        res += constint + 'samplesForMix = ' + str(self.samplesForMix) + ';\n\n'
        
        res += constfloat + 'b0 = ' + str(self.b0) + ';\n'
        res += constfloat + 'db = ' + str(self.db) + ';\n'
        res += constint + 'ndbs = ' + str(self.ndbs) + ';\n\n'
        
        if self.overrideSeed:
            res += 'const uint64_t seed = ' + str(self.newSeed) + ';\n\n'
        else:
            res += 'const uint64_t seed = std::chrono::steady_clock::now().time_since_epoch().count();\n\n'
        
        res += string + 'outDir = \"' + self.outDir + '\";\n'
        res += string + 'outFile = \"' + self.outFile + '\";\n\n'
        
        res += constbool + 'integrateRMS = ' + str(self.IntegrateRMS).lower() + ';\n'
        res += constbool + 'integrateSAB = ' + str(self.IntegrateSAB).lower() + ';\n'
        res += constbool + 'integrateIAB = ' + str(self.IntegrateIAB).lower() + ';\n'
        res += constbool + 'integrateJAB = ' + str(self.IntegrateJAB).lower() + ';\n'
        
        res += "#endif\n"

        return res
        
    def GenerateMacroFile(self) -> None:
        if not os.path.exists(self.macroOutDir + self.macroName + '/'):
            os.mkdir(self.macroOutDir + self.macroName + '/')
        with open(self.macroOutDir + self.macroName + '/' + self.macroName + '.mac', 'w') as file:
            file.write(self.GenerateCode())
        pass
    
    def SetCMakeFileLocation(self, cmakeFile : str) -> None:
        self.cmakeFileLocation = cmakeFile
        pass
    
    def Compile(self) -> None:
        buildDir = self.macroOutDir + self.macroName + '/' + self.macroName + '.build/'
        if os.path.exists(buildDir):
            shutil.rmtree(buildDir)
        os.mkdir(buildDir)

        cmd = 'cd ' + buildDir + ' && '
        cmd += 'cmake -DUSE_MACRO_FILE=ON -DMACRO_FILE_PATH=\\\"'
        cmd += self.macroOutDir + self.macroName + '/' + self.macroName + '.mac'
        cmd += '\\\" ' + self.cmakeFileLocation
        os.system(cmd)
        os.system('cd ' + buildDir + ' && make')
        pass
    
    def Run(self) -> None:
        exe = self.macroOutDir + self.macroOutFile + '.build/main'
        os.system(exe)
        pass