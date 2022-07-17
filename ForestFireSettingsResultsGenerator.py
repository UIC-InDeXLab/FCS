import csv
import pickle
import numpy as np

def CalculateForestFireValues():
    myFile = open('Continuing-Forest-Fire-Settings-Results.csv', 'w')
    myWriter = csv.writer(myFile, delimiter=',')
    
    for k in [2,3,4,5]:
        for npi in [200,400,600,800]:
            contentSpread = []
            disparity = []
            time = []
            initialDisparity = []
            for trial in range(5):
                results = pickle.load(open('forest-fire9-k'+str(k)+'-npi-'+str(npi)+'-trial-'+str(trial)+'.pickle', 'rb'))
                initialDisparity.append(results['Initial Disparity'])
                contentSpread.append(results['ForestFire'][-1]['Lift'])
                disparity.append(results['ForestFire'][-1]['Disparity'])
                time.append(results['ForestFire'][-1]['Time'])
            averageContentSpread = np.mean(contentSpread)
            contentSpreadStandardDeviation = np.std(contentSpread)
            averageDisparity = np.mean(disparity)
            disparityStandardDeviation = np.std(disparity)
            averageTime = np.mean(time)
            timeStandardDeviation = np.std(time)
            averageInitialDisparity = np.mean(initialDisparity)
            initialDisparityStandardDeviation = np.std(initialDisparity)

            myWriter.writerow([k, npi, averageInitialDisparity, initialDisparityStandardDeviation, averageContentSpread, contentSpreadStandardDeviation, averageDisparity, disparityStandardDeviation, averageTime, timeStandardDeviation])

def CalculateVaryingKValues(oneThousand=False):
    myFile = open('Varying-K-Results'+('-1000' if oneThousand else '')+'.csv', 'w')
    myWriter = csv.writer(myFile, delimiter=',')
    
    for k in [2,3,4,5]:
        contentSpread = []
        disparity = []
        time = []
        initialDisparity = []
        cgContentSpread = []
        cgDisparity = []
        cgTime = []
        irfaContentSpread = []
        irfaDisparity = []
        irfaTime = []
        spgreedyContentSpread = []
        spgreedyDisparity = []
        spgreedyTime = []
        acrContentSpread = []
        acrDisparity = []
        acrTime = []
        contentSpreadCommunity = []
        disparityCommunity = []
        timeCommunity = []
        cgContentSpreadCommunity = []
        cgDisparityCommunity = []
        cgTimeCommunity = []
        irfaContentSpreadCommunity = []
        irfaDisparityCommunity = []
        irfaTimeCommunity = []
        spgreedyContentSpreadCommunity = []
        spgreedyDisparityCommunity = []
        spgreedyTimeCommunity = []
        acrContentSpreadCommunity = []
        acrDisparityCommunity = []
        acrTimeCommunity = []
        i = 0
        j = 0
        for instance in range(20):
            fofResults = pickle.load(
                open('varying'+('-1000' if oneThousand else '')+'-k-' + str(k) + '-trial-' + str(instance) + '-fof.pickle', 'rb'))
            communityResults = pickle.load(
                open('varying'+('-1000' if oneThousand else '')+'-k-' + str(k) + '-trial-' + str(instance) + '-community.pickle', 'rb'))
            initialDisparity.append(fofResults['Initial Disparity'])

            if fofResults['Lift'] == 0:
                i = i + 1
                print(i)
            if communityResults['Lift'] == 0:
                j = j + 1
                print(j)
            contentSpread.append(fofResults['Lift'])
            disparity.append(fofResults['Final Disparity'])
            time.append(fofResults['Time'])
            cgContentSpread.append(fofResults['CG Lift'])
            cgDisparity.append(fofResults['CG Final Disparity'])
            cgTime.append(fofResults['CG Time'])
            irfaContentSpread.append(fofResults['IRFA Lift'])
            spgreedyContentSpread.append(fofResults['SPGREEDY Lift'])
            acrContentSpread.append(fofResults['ACR Lift'])
            irfaDisparity.append(fofResults['IRFA Final Disparity'])
            spgreedyDisparity.append(fofResults['SPGREEDY Final Disparity'])
            acrDisparity.append(fofResults['ACR Final Disparity'])
            irfaTime.append(fofResults['IRFA Time'])
            spgreedyTime.append(fofResults['SPGREEDY Time'])
            acrTime.append(fofResults['ACR Time'])

            contentSpreadCommunity.append(communityResults['Lift'])
            disparityCommunity.append(communityResults['Final Disparity'])
            timeCommunity.append(communityResults['Time'])
            cgContentSpreadCommunity.append(communityResults['CG Lift'])
            cgDisparityCommunity.append(communityResults['CG Final Disparity'])
            cgTimeCommunity.append(communityResults['CG Time'])
            irfaContentSpreadCommunity.append(communityResults['IRFA Lift'])
            spgreedyContentSpreadCommunity.append(communityResults['SPGREEDY Lift'])
            acrContentSpreadCommunity.append(communityResults['ACR Lift'])
            irfaDisparityCommunity.append(communityResults['IRFA Final Disparity'])
            spgreedyDisparityCommunity.append(communityResults['SPGREEDY Final Disparity'])
            acrDisparityCommunity.append(communityResults['ACR Final Disparity'])
            irfaTimeCommunity.append(communityResults['IRFA Time'])
            spgreedyTimeCommunity.append(communityResults['SPGREEDY Time'])
            acrTimeCommunity.append(communityResults['ACR Time'])

        averageInitialDisparity = np.mean(initialDisparity)
        initialDisparityStandardDeviation = np.std(initialDisparity)

        averageContentSpread = np.mean(contentSpread)
        contentSpreadStandardDeviation = np.std(contentSpread)
        averageDisparity = np.mean(disparity)
        disparityStandardDeviation = np.std(disparity)
        averageTime = np.mean(time)
        timeStandardDeviation = np.std(time)
        averageCGContentSpread = np.mean(cgContentSpread)
        cgContentSpreadStandardDeviation = np.std(cgContentSpread)
        averageCGDisparity = np.mean(cgDisparity)
        cgDisparityStandardDeviation = np.std(cgDisparity)
        averageCGTime = np.mean(cgTime)
        cgTimeStandardDeviation = np.std(cgTime)
        irfaAverageContentSpread = np.mean(irfaContentSpread)
        irfaContentSpreadStandardDeviation = np.std(irfaContentSpread)
        irfaAverageDisparity = np.mean(irfaDisparity)
        irfaDisparityStandardDeviation = np.std(irfaDisparity)
        irfaAverageTime = np.mean(irfaTime)
        irfaTimeStandardDeviation = np.std(irfaTime)
        spgreedyAverageContentSpread = np.mean(spgreedyContentSpread)
        spgreedyContentSpreadStandardDeviation = np.std(spgreedyContentSpread)
        spgreedyAverageDisparity = np.mean(spgreedyDisparity)
        spgreedyDisparityStandardDeviation = np.std(spgreedyDisparity)
        spgreedyAverageTime = np.mean(spgreedyTime)
        spgreedyTimeStandardDeviation = np.std(spgreedyTime)
        acrAverageContentSpread = np.mean(acrContentSpread)
        acrContentSpreadStandardDeviation = np.std(acrContentSpread)
        acrAverageDisparity = np.mean(acrDisparity)
        acrDisparityStandardDeviation = np.std(acrDisparity)
        acrAverageTime = np.mean(acrTime)
        acrTimeStandardDeviation = np.std(acrTime)

        averageContentSpreadCommunity = np.mean(contentSpreadCommunity)
        contentSpreadStandardDeviationCommunity = np.std(contentSpreadCommunity)
        averageDisparityCommunity = np.mean(disparityCommunity)
        disparityStandardDeviationCommunity = np.std(disparityCommunity)
        averageTimeCommunity = np.mean(timeCommunity)
        timeStandardDeviationCommunity = np.std(timeCommunity)
        averageCGContentSpreadCommunity = np.mean(cgContentSpreadCommunity)
        cgContentSpreadStandardDeviationCommunity = np.std(cgContentSpreadCommunity)
        averageCGDisparityCommunity = np.mean(cgDisparityCommunity)
        cgDisparityStandardDeviationCommunity = np.std(cgDisparityCommunity)
        averageCGTimeCommunity = np.mean(cgTimeCommunity)
        cgTimeStandardDeviationCommunity = np.std(cgTimeCommunity)
        irfaAverageContentSpreadCommunity = np.mean(irfaContentSpreadCommunity)
        irfaContentSpreadStandardDeviationCommunity = np.std(irfaContentSpreadCommunity)
        irfaAverageDisparityCommunity = np.mean(irfaDisparityCommunity)
        irfaDisparityStandardDeviationCommunity = np.std(irfaDisparityCommunity)
        irfaAverageTimeCommunity = np.mean(irfaTimeCommunity)
        irfaTimeStandardDeviationCommunity = np.std(irfaTimeCommunity)
        spgreedyAverageContentSpreadCommunity = np.mean(spgreedyContentSpreadCommunity)
        spgreedyContentSpreadStandardDeviationCommunity = np.std(spgreedyContentSpreadCommunity)
        spgreedyAverageDisparityCommunity = np.mean(spgreedyDisparityCommunity)
        spgreedyDisparityStandardDeviationCommunity = np.std(spgreedyDisparityCommunity)
        spgreedyAverageTimeCommunity = np.mean(spgreedyTimeCommunity)
        spgreedyTimeStandardDeviationCommunity = np.std(spgreedyTimeCommunity)
        acrAverageContentSpreadCommunity = np.mean(acrContentSpreadCommunity)
        acrContentSpreadStandardDeviationCommunity = np.std(acrContentSpreadCommunity)
        acrAverageDisparityCommunity = np.mean(acrDisparityCommunity)
        acrDisparityStandardDeviationCommunity = np.std(acrDisparityCommunity)
        acrAverageTimeCommunity = np.mean(acrTimeCommunity)
        acrTimeStandardDeviationCommunity = np.std(acrTimeCommunity)

        myWriter.writerow([k, averageInitialDisparity, initialDisparityStandardDeviation, averageContentSpread, contentSpreadStandardDeviation, averageDisparity, disparityStandardDeviation, averageTime, timeStandardDeviation, averageCGContentSpread, cgContentSpreadStandardDeviation, averageCGDisparity, cgDisparityStandardDeviation, averageCGTime, cgTimeStandardDeviation, irfaAverageContentSpread, irfaContentSpreadStandardDeviation, irfaAverageDisparity, irfaDisparityStandardDeviation, irfaAverageTime, irfaTimeStandardDeviation, spgreedyAverageContentSpread, spgreedyContentSpreadStandardDeviation, spgreedyAverageDisparity, spgreedyDisparityStandardDeviation, spgreedyAverageTime, spgreedyTimeStandardDeviation, acrAverageContentSpread, acrContentSpreadStandardDeviation, acrAverageDisparity, acrDisparityStandardDeviation, acrAverageTime, acrTimeStandardDeviation, averageContentSpreadCommunity, contentSpreadStandardDeviationCommunity, averageDisparityCommunity, disparityStandardDeviationCommunity, averageTimeCommunity, timeStandardDeviationCommunity, averageCGContentSpreadCommunity, cgContentSpreadStandardDeviationCommunity, averageCGDisparityCommunity, cgDisparityStandardDeviationCommunity, averageCGTimeCommunity, cgTimeStandardDeviationCommunity, irfaAverageContentSpreadCommunity, irfaContentSpreadStandardDeviationCommunity, irfaAverageDisparityCommunity, irfaDisparityStandardDeviationCommunity, irfaAverageTimeCommunity, irfaTimeStandardDeviationCommunity, spgreedyAverageContentSpreadCommunity, spgreedyContentSpreadStandardDeviationCommunity, spgreedyAverageDisparityCommunity, spgreedyDisparityStandardDeviationCommunity, spgreedyAverageTimeCommunity, spgreedyTimeStandardDeviationCommunity, acrAverageContentSpreadCommunity, acrContentSpreadStandardDeviationCommunity, acrAverageDisparityCommunity, acrDisparityStandardDeviationCommunity, acrAverageTimeCommunity, acrTimeStandardDeviationCommunity])

def CalculateVaryingKValuesMultiFile(oneThousand=False):
    liftFile = open('Varying-K-Results-Lift'+('-1000' if oneThousand else '')+'.csv', 'w')
    fairFile = open('Varying-K-Results-Fair'+('-1000' if oneThousand else '')+'.csv', 'w')
    timeFile = open('Varying-K-Results-Time' + ('-1000' if oneThousand else '') + '.csv', 'w')

    liftWriter = csv.writer(liftFile, delimiter=',')
    fairWriter = csv.writer(fairFile, delimiter=',')
    timeWriter = csv.writer(timeFile, delimiter=',')


    for k in [2,3,4,5]:
        contentSpread = []
        disparity = []
        time = []
        initialDisparity = []
        cgContentSpread = []
        cgDisparity = []
        cgTime = []
        irfaContentSpread = []
        irfaDisparity = []
        irfaTime = []
        spgreedyContentSpread = []
        spgreedyDisparity = []
        spgreedyTime = []
        acrContentSpread = []
        acrDisparity = []
        acrTime = []
        contentSpreadCommunity = []
        disparityCommunity = []
        timeCommunity = []
        cgContentSpreadCommunity = []
        cgDisparityCommunity = []
        cgTimeCommunity = []
        irfaContentSpreadCommunity = []
        irfaDisparityCommunity = []
        irfaTimeCommunity = []
        spgreedyContentSpreadCommunity = []
        spgreedyDisparityCommunity = []
        spgreedyTimeCommunity = []
        acrContentSpreadCommunity = []
        acrDisparityCommunity = []
        acrTimeCommunity = []
        i = 0
        j = 0
        for instance in range(20):
            fofResults = pickle.load(
                open('varying'+('-1000' if oneThousand else '')+'-k-' + str(k) + '-trial-' + str(instance) + '-fof.pickle', 'rb'))
            communityResults = pickle.load(
                open('varying'+('-1000' if oneThousand else '')+'-k-' + str(k) + '-trial-' + str(instance) + '-community.pickle', 'rb'))
            initialDisparity.append(fofResults['Initial Disparity'])

            if fofResults['Lift'] == 0:
                i = i + 1
                print(i)
            if communityResults['Lift'] == 0:
                j = j + 1
                print(j)
            contentSpread.append(fofResults['Lift'])
            disparity.append(fofResults['Final Disparity'])
            time.append(fofResults['Time'])
            cgContentSpread.append(fofResults['CG Lift'])
            cgDisparity.append(fofResults['CG Final Disparity'])
            cgTime.append(fofResults['CG Time'])
            irfaContentSpread.append(fofResults['IRFA Lift'])
            spgreedyContentSpread.append(fofResults['SPGREEDY Lift'])
            acrContentSpread.append(fofResults['ACR Lift'])
            irfaDisparity.append(fofResults['IRFA Final Disparity'])
            spgreedyDisparity.append(fofResults['SPGREEDY Final Disparity'])
            acrDisparity.append(fofResults['ACR Final Disparity'])
            irfaTime.append(fofResults['IRFA Time'])
            spgreedyTime.append(fofResults['SPGREEDY Time'])
            acrTime.append(fofResults['ACR Time'])

            contentSpreadCommunity.append(communityResults['Lift'])
            disparityCommunity.append(communityResults['Final Disparity'])
            timeCommunity.append(communityResults['Time'])
            cgContentSpreadCommunity.append(communityResults['CG Lift'])
            cgDisparityCommunity.append(communityResults['CG Final Disparity'])
            cgTimeCommunity.append(communityResults['CG Time'])
            irfaContentSpreadCommunity.append(communityResults['IRFA Lift'])
            spgreedyContentSpreadCommunity.append(communityResults['SPGREEDY Lift'])
            acrContentSpreadCommunity.append(communityResults['ACR Lift'])
            irfaDisparityCommunity.append(communityResults['IRFA Final Disparity'])
            spgreedyDisparityCommunity.append(communityResults['SPGREEDY Final Disparity'])
            acrDisparityCommunity.append(communityResults['ACR Final Disparity'])
            irfaTimeCommunity.append(communityResults['IRFA Time'])
            spgreedyTimeCommunity.append(communityResults['SPGREEDY Time'])
            acrTimeCommunity.append(communityResults['ACR Time'])

        averageInitialDisparity = np.mean(initialDisparity)
        initialDisparityStandardDeviation = np.std(initialDisparity)

        averageContentSpread = np.mean(contentSpread)
        contentSpreadStandardDeviation = np.std(contentSpread)
        averageDisparity = np.mean(disparity)
        disparityStandardDeviation = np.std(disparity)
        averageTime = np.mean(time)
        timeStandardDeviation = np.std(time)
        averageCGContentSpread = np.mean(cgContentSpread)
        cgContentSpreadStandardDeviation = np.std(cgContentSpread)
        averageCGDisparity = np.mean(cgDisparity)
        cgDisparityStandardDeviation = np.std(cgDisparity)
        averageCGTime = np.mean(cgTime)
        cgTimeStandardDeviation = np.std(cgTime)
        irfaAverageContentSpread = np.mean(irfaContentSpread)
        irfaContentSpreadStandardDeviation = np.std(irfaContentSpread)
        irfaAverageDisparity = np.mean(irfaDisparity)
        irfaDisparityStandardDeviation = np.std(irfaDisparity)
        irfaAverageTime = np.mean(irfaTime)
        irfaTimeStandardDeviation = np.std(irfaTime)
        spgreedyAverageContentSpread = np.mean(spgreedyContentSpread)
        spgreedyContentSpreadStandardDeviation = np.std(spgreedyContentSpread)
        spgreedyAverageDisparity = np.mean(spgreedyDisparity)
        spgreedyDisparityStandardDeviation = np.std(spgreedyDisparity)
        spgreedyAverageTime = np.mean(spgreedyTime)
        spgreedyTimeStandardDeviation = np.std(spgreedyTime)
        acrAverageContentSpread = np.mean(acrContentSpread)
        acrContentSpreadStandardDeviation = np.std(acrContentSpread)
        acrAverageDisparity = np.mean(acrDisparity)
        acrDisparityStandardDeviation = np.std(acrDisparity)
        acrAverageTime = np.mean(acrTime)
        acrTimeStandardDeviation = np.std(acrTime)

        averageContentSpreadCommunity = np.mean(contentSpreadCommunity)
        contentSpreadStandardDeviationCommunity = np.std(contentSpreadCommunity)
        averageDisparityCommunity = np.mean(disparityCommunity)
        disparityStandardDeviationCommunity = np.std(disparityCommunity)
        averageTimeCommunity = np.mean(timeCommunity)
        timeStandardDeviationCommunity = np.std(timeCommunity)
        averageCGContentSpreadCommunity = np.mean(cgContentSpreadCommunity)
        cgContentSpreadStandardDeviationCommunity = np.std(cgContentSpreadCommunity)
        averageCGDisparityCommunity = np.mean(cgDisparityCommunity)
        cgDisparityStandardDeviationCommunity = np.std(cgDisparityCommunity)
        averageCGTimeCommunity = np.mean(cgTimeCommunity)
        cgTimeStandardDeviationCommunity = np.std(cgTimeCommunity)
        irfaAverageContentSpreadCommunity = np.mean(irfaContentSpreadCommunity)
        irfaContentSpreadStandardDeviationCommunity = np.std(irfaContentSpreadCommunity)
        irfaAverageDisparityCommunity = np.mean(irfaDisparityCommunity)
        irfaDisparityStandardDeviationCommunity = np.std(irfaDisparityCommunity)
        irfaAverageTimeCommunity = np.mean(irfaTimeCommunity)
        irfaTimeStandardDeviationCommunity = np.std(irfaTimeCommunity)
        spgreedyAverageContentSpreadCommunity = np.mean(spgreedyContentSpreadCommunity)
        spgreedyContentSpreadStandardDeviationCommunity = np.std(spgreedyContentSpreadCommunity)
        spgreedyAverageDisparityCommunity = np.mean(spgreedyDisparityCommunity)
        spgreedyDisparityStandardDeviationCommunity = np.std(spgreedyDisparityCommunity)
        spgreedyAverageTimeCommunity = np.mean(spgreedyTimeCommunity)
        spgreedyTimeStandardDeviationCommunity = np.std(spgreedyTimeCommunity)
        acrAverageContentSpreadCommunity = np.mean(acrContentSpreadCommunity)
        acrContentSpreadStandardDeviationCommunity = np.std(acrContentSpreadCommunity)
        acrAverageDisparityCommunity = np.mean(acrDisparityCommunity)
        acrDisparityStandardDeviationCommunity = np.std(acrDisparityCommunity)
        acrAverageTimeCommunity = np.mean(acrTimeCommunity)
        acrTimeStandardDeviationCommunity = np.std(acrTimeCommunity)


        liftWriter.writerow([k, round(averageContentSpread, 2), round(contentSpreadStandardDeviation, 2),  round(averageCGContentSpread, 2), round(cgContentSpreadStandardDeviation, 2),
                           round(irfaAverageContentSpread, 2), round(irfaContentSpreadStandardDeviation, 2), round(spgreedyAverageContentSpread, 2), round(spgreedyContentSpreadStandardDeviation, 2),
                           round(acrAverageContentSpread, 2), round(acrContentSpreadStandardDeviation, 2)])
        liftWriter.writerow([k, round(averageContentSpreadCommunity, 2), round(contentSpreadStandardDeviationCommunity, 2),
                           round(averageCGContentSpreadCommunity, 2), round(cgContentSpreadStandardDeviationCommunity, 2), round(irfaAverageContentSpreadCommunity, 2),
                            round(irfaContentSpreadStandardDeviationCommunity, 2), round(spgreedyAverageContentSpreadCommunity, 2),
                           round(spgreedyContentSpreadStandardDeviationCommunity, 2), round(acrAverageContentSpreadCommunity, 2),
                           round(acrContentSpreadStandardDeviationCommunity, 2)])

        fairWriter.writerow([k, round(averageInitialDisparity, 2), round(initialDisparityStandardDeviation, 2), round(averageDisparity, 2), round(disparityStandardDeviation, 2),
                           round(averageCGDisparity, 2), round(cgDisparityStandardDeviation, 2),  round(irfaAverageDisparity, 2),
                           round(irfaDisparityStandardDeviation, 2),round(spgreedyAverageDisparity, 2), round(spgreedyDisparityStandardDeviation, 2),
                           round(acrAverageDisparity, 2), round(acrDisparityStandardDeviation, 2)])
        fairWriter.writerow([k, round(averageInitialDisparity, 2), round(initialDisparityStandardDeviation, 2),
                           round(averageDisparityCommunity, 2), round(disparityStandardDeviationCommunity, 2), round(averageCGDisparityCommunity, 2),
                           round(cgDisparityStandardDeviationCommunity, 2), round(irfaAverageDisparityCommunity, 2),
                           round(irfaDisparityStandardDeviationCommunity, 2), round(spgreedyAverageDisparityCommunity, 2),
                           round(spgreedyDisparityStandardDeviationCommunity, 2), round(acrAverageDisparityCommunity, 2),
                           round(acrDisparityStandardDeviationCommunity, 2)])

        timeWriter.writerow([k, round(averageTime, 2), round(timeStandardDeviation, 2), round(averageCGTime, 2), round(cgTimeStandardDeviation, 2),
                            round(irfaAverageTime, 2), round(irfaTimeStandardDeviation, 2), round(spgreedyAverageTime, 2), round(spgreedyTimeStandardDeviation, 2),
                             round(acrAverageTime, 2), round(acrTimeStandardDeviation, 2)])
        timeWriter.writerow([k, round(averageTimeCommunity, 2), round(timeStandardDeviationCommunity, 2),
                           round(averageCGTimeCommunity, 2), round(cgTimeStandardDeviationCommunity, 2), round(irfaAverageTimeCommunity, 2),
                           round(irfaTimeStandardDeviationCommunity, 2), round(spgreedyAverageTimeCommunity, 2),
                           round(spgreedyTimeStandardDeviationCommunity, 2), round(acrAverageTimeCommunity, 2),
                           round(acrTimeStandardDeviationCommunity, 2)])

def CalculateVaryingPValues(oneThousand=False):
    myFile = open('Varying-P-Results'+('-1000' if oneThousand else '')+'.csv', 'w')
    myWriter = csv.writer(myFile, delimiter=',')

    for p in [0.3, 0.5, 0.7, 0.9]:
        contentSpread = []
        disparity = []
        time = []
        initialDisparity = []
        cgContentSpread = []
        cgDisparity = []
        cgTime = []
        irfaContentSpread = []
        irfaDisparity = []
        irfaTime = []
        spgreedyContentSpread = []
        spgreedyDisparity = []
        spgreedyTime = []
        acrContentSpread = []
        acrDisparity = []
        acrTime = []
        contentSpreadCommunity = []
        disparityCommunity = []
        timeCommunity = []
        cgContentSpreadCommunity = []
        cgDisparityCommunity = []
        cgTimeCommunity = []
        irfaContentSpreadCommunity = []
        irfaDisparityCommunity = []
        irfaTimeCommunity = []
        spgreedyContentSpreadCommunity = []
        spgreedyDisparityCommunity = []
        spgreedyTimeCommunity = []
        acrContentSpreadCommunity = []
        acrDisparityCommunity = []
        acrTimeCommunity = []
        i = 0
        j = 0
        for instance in range(20):
            fofResults = pickle.load(
                open('varying'+('-1000' if oneThousand else '')+'-p-' + str(p) + '-trial-' + str(instance) + '-fof.pickle', 'rb'))
            communityResults = pickle.load(
                open('varying'+('-1000' if oneThousand else '')+'-p-' + str(p) + '-trial-' + str(instance) + '-community.pickle', 'rb'))
            initialDisparity.append(fofResults['Initial Disparity'])

            if fofResults['Lift'] == 0:
                i = i + 1
                print(i)
            if communityResults['Lift'] == 0:
                j = j + 1
                print(j)

            contentSpread.append(fofResults['Lift'])
            disparity.append(fofResults['Final Disparity'])
            time.append(fofResults['Time'])
            cgContentSpread.append(fofResults['CG Lift'])
            cgDisparity.append(fofResults['CG Final Disparity'])
            cgTime.append(fofResults['CG Time'])
            irfaContentSpread.append(fofResults['IRFA Lift'])
            spgreedyContentSpread.append(fofResults['SPGREEDY Lift'])
            acrContentSpread.append(fofResults['ACR Lift'])
            irfaDisparity.append(fofResults['IRFA Final Disparity'])
            spgreedyDisparity.append(fofResults['SPGREEDY Final Disparity'])
            acrDisparity.append(fofResults['ACR Final Disparity'])
            irfaTime.append(fofResults['IRFA Time'])
            spgreedyTime.append(fofResults['SPGREEDY Time'])
            acrTime.append(fofResults['ACR Time'])

            contentSpreadCommunity.append(communityResults['Lift'])
            disparityCommunity.append(communityResults['Final Disparity'])
            timeCommunity.append(communityResults['Time'])
            cgContentSpreadCommunity.append(communityResults['CG Lift'])
            cgDisparityCommunity.append(communityResults['CG Final Disparity'])
            cgTimeCommunity.append(communityResults['CG Time'])
            irfaContentSpreadCommunity.append(communityResults['IRFA Lift'])
            spgreedyContentSpreadCommunity.append(communityResults['SPGREEDY Lift'])
            acrContentSpreadCommunity.append(communityResults['ACR Lift'])
            irfaDisparityCommunity.append(communityResults['IRFA Final Disparity'])
            spgreedyDisparityCommunity.append(communityResults['SPGREEDY Final Disparity'])
            acrDisparityCommunity.append(communityResults['ACR Final Disparity'])
            irfaTimeCommunity.append(communityResults['IRFA Time'])
            spgreedyTimeCommunity.append(communityResults['SPGREEDY Time'])
            acrTimeCommunity.append(communityResults['ACR Time'])

        averageInitialDisparity = np.mean(initialDisparity)
        initialDisparityStandardDeviation = np.std(initialDisparity)

        averageContentSpread = np.mean(contentSpread)
        contentSpreadStandardDeviation = np.std(contentSpread)
        averageDisparity = np.mean(disparity)
        disparityStandardDeviation = np.std(disparity)
        averageTime = np.mean(time)
        timeStandardDeviation = np.std(time)
        averageCGContentSpread = np.mean(cgContentSpread)
        cgContentSpreadStandardDeviation = np.std(cgContentSpread)
        averageCGDisparity = np.mean(cgDisparity)
        cgDisparityStandardDeviation = np.std(cgDisparity)
        averageCGTime = np.mean(cgTime)
        cgTimeStandardDeviation = np.std(cgTime)
        irfaAverageContentSpread = np.mean(irfaContentSpread)
        irfaContentSpreadStandardDeviation = np.std(irfaContentSpread)
        irfaAverageDisparity = np.mean(irfaDisparity)
        irfaDisparityStandardDeviation = np.std(irfaDisparity)
        irfaAverageTime = np.mean(irfaTime)
        irfaTimeStandardDeviation = np.std(irfaTime)
        spgreedyAverageContentSpread = np.mean(spgreedyContentSpread)
        spgreedyContentSpreadStandardDeviation = np.std(spgreedyContentSpread)
        spgreedyAverageDisparity = np.mean(spgreedyDisparity)
        spgreedyDisparityStandardDeviation = np.std(spgreedyDisparity)
        spgreedyAverageTime = np.mean(spgreedyTime)
        spgreedyTimeStandardDeviation = np.std(spgreedyTime)
        acrAverageContentSpread = np.mean(acrContentSpread)
        acrContentSpreadStandardDeviation = np.std(acrContentSpread)
        acrAverageDisparity = np.mean(acrDisparity)
        acrDisparityStandardDeviation = np.std(acrDisparity)
        acrAverageTime = np.mean(acrTime)
        acrTimeStandardDeviation = np.std(acrTime)

        averageContentSpreadCommunity = np.mean(contentSpreadCommunity)
        contentSpreadStandardDeviationCommunity = np.std(contentSpreadCommunity)
        averageDisparityCommunity = np.mean(disparityCommunity)
        disparityStandardDeviationCommunity = np.std(disparityCommunity)
        averageTimeCommunity = np.mean(timeCommunity)
        timeStandardDeviationCommunity = np.std(timeCommunity)
        averageCGContentSpreadCommunity = np.mean(cgContentSpreadCommunity)
        cgContentSpreadStandardDeviationCommunity = np.std(cgContentSpreadCommunity)
        averageCGDisparityCommunity = np.mean(cgDisparityCommunity)
        cgDisparityStandardDeviationCommunity = np.std(cgDisparityCommunity)
        averageCGTimeCommunity = np.mean(cgTimeCommunity)
        cgTimeStandardDeviationCommunity = np.std(cgTimeCommunity)
        irfaAverageContentSpreadCommunity = np.mean(irfaContentSpreadCommunity)
        irfaContentSpreadStandardDeviationCommunity = np.std(irfaContentSpreadCommunity)
        irfaAverageDisparityCommunity = np.mean(irfaDisparityCommunity)
        irfaDisparityStandardDeviationCommunity = np.std(irfaDisparityCommunity)
        irfaAverageTimeCommunity = np.mean(irfaTimeCommunity)
        irfaTimeStandardDeviationCommunity = np.std(irfaTimeCommunity)
        spgreedyAverageContentSpreadCommunity = np.mean(spgreedyContentSpreadCommunity)
        spgreedyContentSpreadStandardDeviationCommunity = np.std(spgreedyContentSpreadCommunity)
        spgreedyAverageDisparityCommunity = np.mean(spgreedyDisparityCommunity)
        spgreedyDisparityStandardDeviationCommunity = np.std(spgreedyDisparityCommunity)
        spgreedyAverageTimeCommunity = np.mean(spgreedyTimeCommunity)
        spgreedyTimeStandardDeviationCommunity = np.std(spgreedyTimeCommunity)
        acrAverageContentSpreadCommunity = np.mean(acrContentSpreadCommunity)
        acrContentSpreadStandardDeviationCommunity = np.std(acrContentSpreadCommunity)
        acrAverageDisparityCommunity = np.mean(acrDisparityCommunity)
        acrDisparityStandardDeviationCommunity = np.std(acrDisparityCommunity)
        acrAverageTimeCommunity = np.mean(acrTimeCommunity)
        acrTimeStandardDeviationCommunity = np.std(acrTimeCommunity)

        myWriter.writerow([p, averageInitialDisparity, initialDisparityStandardDeviation, averageContentSpread,
                           contentSpreadStandardDeviation, averageDisparity, disparityStandardDeviation, averageTime,
                           timeStandardDeviation, averageCGContentSpread, cgContentSpreadStandardDeviation,
                           averageCGDisparity, cgDisparityStandardDeviation, averageCGTime, cgTimeStandardDeviation,
                           irfaAverageContentSpread, irfaContentSpreadStandardDeviation, irfaAverageDisparity,
                           irfaDisparityStandardDeviation, irfaAverageTime, irfaTimeStandardDeviation,
                           spgreedyAverageContentSpread, spgreedyContentSpreadStandardDeviation,
                           spgreedyAverageDisparity, spgreedyDisparityStandardDeviation, spgreedyAverageTime,
                           spgreedyTimeStandardDeviation, acrAverageContentSpread, acrContentSpreadStandardDeviation,
                           acrAverageDisparity, acrDisparityStandardDeviation, acrAverageTime, acrTimeStandardDeviation,
                           averageContentSpreadCommunity, contentSpreadStandardDeviationCommunity,
                           averageDisparityCommunity, disparityStandardDeviationCommunity, averageTimeCommunity,
                           timeStandardDeviationCommunity, averageCGContentSpreadCommunity,
                           cgContentSpreadStandardDeviationCommunity, averageCGDisparityCommunity,
                           cgDisparityStandardDeviationCommunity, averageCGTimeCommunity,
                           cgTimeStandardDeviationCommunity, irfaAverageContentSpreadCommunity,
                           irfaContentSpreadStandardDeviationCommunity, irfaAverageDisparityCommunity,
                           irfaDisparityStandardDeviationCommunity, irfaAverageTimeCommunity,
                           irfaTimeStandardDeviationCommunity, spgreedyAverageContentSpreadCommunity,
                           spgreedyContentSpreadStandardDeviationCommunity, spgreedyAverageDisparityCommunity,
                           spgreedyDisparityStandardDeviationCommunity, spgreedyAverageTimeCommunity,
                           spgreedyTimeStandardDeviationCommunity, acrAverageContentSpreadCommunity,
                           acrContentSpreadStandardDeviationCommunity, acrAverageDisparityCommunity,
                           acrDisparityStandardDeviationCommunity, acrAverageTimeCommunity,
                           acrTimeStandardDeviationCommunity])



def CalculateVaryingPValuesMultiFile(oneThousand=False):
    liftFile = open('Varying-P-Results-Lift'+('-1000' if oneThousand else '')+'.csv', 'w')
    fairFile = open('Varying-P-Results-Fair'+('-1000' if oneThousand else '')+'.csv', 'w')
    timeFile = open('Varying-P-Results-Time' + ('-1000' if oneThousand else '') + '.csv', 'w')

    liftWriter = csv.writer(liftFile, delimiter=',')
    fairWriter = csv.writer(fairFile, delimiter=',')
    timeWriter = csv.writer(timeFile, delimiter=',')

    for p in [0.3, 0.5, 0.7, 0.9]:
        contentSpread = []
        disparity = []
        time = []
        initialDisparity = []
        cgContentSpread = []
        cgDisparity = []
        cgTime = []
        irfaContentSpread = []
        irfaDisparity = []
        irfaTime = []
        spgreedyContentSpread = []
        spgreedyDisparity = []
        spgreedyTime = []
        acrContentSpread = []
        acrDisparity = []
        acrTime = []
        contentSpreadCommunity = []
        disparityCommunity = []
        timeCommunity = []
        cgContentSpreadCommunity = []
        cgDisparityCommunity = []
        cgTimeCommunity = []
        irfaContentSpreadCommunity = []
        irfaDisparityCommunity = []
        irfaTimeCommunity = []
        spgreedyContentSpreadCommunity = []
        spgreedyDisparityCommunity = []
        spgreedyTimeCommunity = []
        acrContentSpreadCommunity = []
        acrDisparityCommunity = []
        acrTimeCommunity = []
        i = 0
        j = 0
        for instance in range(20):
            fofResults = pickle.load(
                open('varying'+('-1000' if oneThousand else '')+'-p-' + str(p) + '-trial-' + str(instance) + '-fof.pickle', 'rb'))
            communityResults = pickle.load(
                open('varying'+('-1000' if oneThousand else '')+'-p-' + str(p) + '-trial-' + str(instance) + '-community.pickle', 'rb'))
            initialDisparity.append(fofResults['Initial Disparity'])

            if fofResults['Lift'] == 0:
                i = i + 1
                print(i)
            if communityResults['Lift'] == 0:
                j = j + 1
                print(j)

            contentSpread.append(fofResults['Lift'])
            disparity.append(fofResults['Final Disparity'])
            time.append(fofResults['Time'])
            cgContentSpread.append(fofResults['CG Lift'])
            cgDisparity.append(fofResults['CG Final Disparity'])
            cgTime.append(fofResults['CG Time'])
            irfaContentSpread.append(fofResults['IRFA Lift'])
            spgreedyContentSpread.append(fofResults['SPGREEDY Lift'])
            acrContentSpread.append(fofResults['ACR Lift'])
            irfaDisparity.append(fofResults['IRFA Final Disparity'])
            spgreedyDisparity.append(fofResults['SPGREEDY Final Disparity'])
            acrDisparity.append(fofResults['ACR Final Disparity'])
            irfaTime.append(fofResults['IRFA Time'])
            spgreedyTime.append(fofResults['SPGREEDY Time'])
            acrTime.append(fofResults['ACR Time'])

            contentSpreadCommunity.append(communityResults['Lift'])
            disparityCommunity.append(communityResults['Final Disparity'])
            timeCommunity.append(communityResults['Time'])
            cgContentSpreadCommunity.append(communityResults['CG Lift'])
            cgDisparityCommunity.append(communityResults['CG Final Disparity'])
            cgTimeCommunity.append(communityResults['CG Time'])
            irfaContentSpreadCommunity.append(communityResults['IRFA Lift'])
            spgreedyContentSpreadCommunity.append(communityResults['SPGREEDY Lift'])
            acrContentSpreadCommunity.append(communityResults['ACR Lift'])
            irfaDisparityCommunity.append(communityResults['IRFA Final Disparity'])
            spgreedyDisparityCommunity.append(communityResults['SPGREEDY Final Disparity'])
            acrDisparityCommunity.append(communityResults['ACR Final Disparity'])
            irfaTimeCommunity.append(communityResults['IRFA Time'])
            spgreedyTimeCommunity.append(communityResults['SPGREEDY Time'])
            acrTimeCommunity.append(communityResults['ACR Time'])

        averageInitialDisparity = np.mean(initialDisparity)
        initialDisparityStandardDeviation = np.std(initialDisparity)

        averageContentSpread = np.mean(contentSpread)
        contentSpreadStandardDeviation = np.std(contentSpread)
        averageDisparity = np.mean(disparity)
        disparityStandardDeviation = np.std(disparity)
        averageTime = np.mean(time)
        timeStandardDeviation = np.std(time)
        averageCGContentSpread = np.mean(cgContentSpread)
        cgContentSpreadStandardDeviation = np.std(cgContentSpread)
        averageCGDisparity = np.mean(cgDisparity)
        cgDisparityStandardDeviation = np.std(cgDisparity)
        averageCGTime = np.mean(cgTime)
        cgTimeStandardDeviation = np.std(cgTime)
        irfaAverageContentSpread = np.mean(irfaContentSpread)
        irfaContentSpreadStandardDeviation = np.std(irfaContentSpread)
        irfaAverageDisparity = np.mean(irfaDisparity)
        irfaDisparityStandardDeviation = np.std(irfaDisparity)
        irfaAverageTime = np.mean(irfaTime)
        irfaTimeStandardDeviation = np.std(irfaTime)
        spgreedyAverageContentSpread = np.mean(spgreedyContentSpread)
        spgreedyContentSpreadStandardDeviation = np.std(spgreedyContentSpread)
        spgreedyAverageDisparity = np.mean(spgreedyDisparity)
        spgreedyDisparityStandardDeviation = np.std(spgreedyDisparity)
        spgreedyAverageTime = np.mean(spgreedyTime)
        spgreedyTimeStandardDeviation = np.std(spgreedyTime)
        acrAverageContentSpread = np.mean(acrContentSpread)
        acrContentSpreadStandardDeviation = np.std(acrContentSpread)
        acrAverageDisparity = np.mean(acrDisparity)
        acrDisparityStandardDeviation = np.std(acrDisparity)
        acrAverageTime = np.mean(acrTime)
        acrTimeStandardDeviation = np.std(acrTime)

        averageContentSpreadCommunity = np.mean(contentSpreadCommunity)
        contentSpreadStandardDeviationCommunity = np.std(contentSpreadCommunity)
        averageDisparityCommunity = np.mean(disparityCommunity)
        disparityStandardDeviationCommunity = np.std(disparityCommunity)
        averageTimeCommunity = np.mean(timeCommunity)
        timeStandardDeviationCommunity = np.std(timeCommunity)
        averageCGContentSpreadCommunity = np.mean(cgContentSpreadCommunity)
        cgContentSpreadStandardDeviationCommunity = np.std(cgContentSpreadCommunity)
        averageCGDisparityCommunity = np.mean(cgDisparityCommunity)
        cgDisparityStandardDeviationCommunity = np.std(cgDisparityCommunity)
        averageCGTimeCommunity = np.mean(cgTimeCommunity)
        cgTimeStandardDeviationCommunity = np.std(cgTimeCommunity)
        irfaAverageContentSpreadCommunity = np.mean(irfaContentSpreadCommunity)
        irfaContentSpreadStandardDeviationCommunity = np.std(irfaContentSpreadCommunity)
        irfaAverageDisparityCommunity = np.mean(irfaDisparityCommunity)
        irfaDisparityStandardDeviationCommunity = np.std(irfaDisparityCommunity)
        irfaAverageTimeCommunity = np.mean(irfaTimeCommunity)
        irfaTimeStandardDeviationCommunity = np.std(irfaTimeCommunity)
        spgreedyAverageContentSpreadCommunity = np.mean(spgreedyContentSpreadCommunity)
        spgreedyContentSpreadStandardDeviationCommunity = np.std(spgreedyContentSpreadCommunity)
        spgreedyAverageDisparityCommunity = np.mean(spgreedyDisparityCommunity)
        spgreedyDisparityStandardDeviationCommunity = np.std(spgreedyDisparityCommunity)
        spgreedyAverageTimeCommunity = np.mean(spgreedyTimeCommunity)
        spgreedyTimeStandardDeviationCommunity = np.std(spgreedyTimeCommunity)
        acrAverageContentSpreadCommunity = np.mean(acrContentSpreadCommunity)
        acrContentSpreadStandardDeviationCommunity = np.std(acrContentSpreadCommunity)
        acrAverageDisparityCommunity = np.mean(acrDisparityCommunity)
        acrDisparityStandardDeviationCommunity = np.std(acrDisparityCommunity)
        acrAverageTimeCommunity = np.mean(acrTimeCommunity)
        acrTimeStandardDeviationCommunity = np.std(acrTimeCommunity)

        liftWriter.writerow([p, round(averageContentSpread, 2), round(contentSpreadStandardDeviation, 2),  round(averageCGContentSpread, 2), round(cgContentSpreadStandardDeviation, 2),
                           round(irfaAverageContentSpread, 2), round(irfaContentSpreadStandardDeviation, 2), round(spgreedyAverageContentSpread, 2), round(spgreedyContentSpreadStandardDeviation, 2),
                           round(acrAverageContentSpread, 2), round(acrContentSpreadStandardDeviation, 2)])
        liftWriter.writerow([p, round(averageContentSpreadCommunity, 2), round(contentSpreadStandardDeviationCommunity, 2),
                           round(averageCGContentSpreadCommunity, 2), round(cgContentSpreadStandardDeviationCommunity, 2), round(irfaAverageContentSpreadCommunity, 2),
                            round(irfaContentSpreadStandardDeviationCommunity, 2), round(spgreedyAverageContentSpreadCommunity, 2),
                           round(spgreedyContentSpreadStandardDeviationCommunity, 2), round(acrAverageContentSpreadCommunity, 2),
                           round(acrContentSpreadStandardDeviationCommunity, 2)])

        fairWriter.writerow([p, round(averageInitialDisparity, 2), round(initialDisparityStandardDeviation, 2), round(averageDisparity, 2), round(disparityStandardDeviation, 2),
                           round(averageCGDisparity, 2), round(cgDisparityStandardDeviation, 2),  round(irfaAverageDisparity, 2),
                           round(irfaDisparityStandardDeviation, 2),round(spgreedyAverageDisparity, 2), round(spgreedyDisparityStandardDeviation, 2),
                           round(acrAverageDisparity, 2), round(acrDisparityStandardDeviation, 2)])
        fairWriter.writerow([p, round(averageInitialDisparity, 2), round(initialDisparityStandardDeviation, 2),
                           round(averageDisparityCommunity, 2), round(disparityStandardDeviationCommunity, 2), round(averageCGDisparityCommunity, 2),
                           round(cgDisparityStandardDeviationCommunity, 2), round(irfaAverageDisparityCommunity, 2),
                           round(irfaDisparityStandardDeviationCommunity, 2), round(spgreedyAverageDisparityCommunity, 2),
                           round(spgreedyDisparityStandardDeviationCommunity, 2), round(acrAverageDisparityCommunity, 2),
                           round(acrDisparityStandardDeviationCommunity, 2)])

        timeWriter.writerow([p, round(averageTime, 2), round(timeStandardDeviation, 2), round(averageCGTime, 2), round(cgTimeStandardDeviation, 2),
                            round(irfaAverageTime, 2), round(irfaTimeStandardDeviation, 2), round(spgreedyAverageTime, 2), round(spgreedyTimeStandardDeviation, 2),
                             round(acrAverageTime, 2), round(acrTimeStandardDeviation, 2)])
        timeWriter.writerow([p, round(averageTimeCommunity, 2), round(timeStandardDeviationCommunity, 2),
                           round(averageCGTimeCommunity, 2), round(cgTimeStandardDeviationCommunity, 2), round(irfaAverageTimeCommunity, 2),
                           round(irfaTimeStandardDeviationCommunity, 2), round(spgreedyAverageTimeCommunity, 2),
                           round(spgreedyTimeStandardDeviationCommunity, 2), round(acrAverageTimeCommunity, 2),
                           round(acrTimeStandardDeviationCommunity, 2)])

def CalculateAlgorithmsComparison():
    myFile = open('Comparison-Results.csv', 'w')
    myWriter = csv.writer(myFile, delimiter=',')
    
    for size in range(5):
        lpAdvanced = []
        continuousGreedy = []
        forestFire = []
        initialDisparity = []

        for trial in range(5):
            if (size < 4):
                cur = pickle.load(open('medium-graph-IterFCS-size-'+str(size)+'-trial-'+str(trial)+'.pickle', 'rb'))
                lpAdvanced.append((cur['IterFCS'][-1]['Disparity'], cur['IterFCS'][-1]['Lift'], cur['IterFCS'][-1]['Time']))
            cur = pickle.load(open('medium-graph-ForestFire-size-'+str(size)+'-trial-'+str(trial)+'.pickle', 'rb'))
            forestFire.append((cur['ForestFire'][-1]['Disparity'], cur['ForestFire'][-1]['Lift'], cur['ForestFire'][-1]['Time']))
            cur = pickle.load(open('medium-graph-CG-size-'+str(size)+'-trial-'+str(trial)+'.pickle', 'rb'))
            continuousGreedy.append((cur['CG']['Disparity'], cur['CG']['CS'], cur['CG']['Time']))
            initialDisparity.append(cur['Initial Disparity'])
        

        row = []
        row.append(np.mean(initialDisparity))
        row.append(np.std(initialDisparity))
        for source in [lpAdvanced, continuousGreedy, forestFire]:
            for i in range(3):
                row.append(np.mean([x[i] for x in source]))
                row.append(np.std([x[i] for x in source]))

        myWriter.writerow(row)


def CalculateAlgorithmsComparison2():
    myFile = open('Comparison-Results-2.csv', 'w')
    myWriter = csv.writer(myFile, delimiter=',')

    for size in range(4):
        irfa = []
        spgreedy = []
        acr = []
        lpAdvanced = []
        forestFire = []
        continuousGreedy = []
        initialDisparity = []

        for trial in range(5):

            if (size < 1):
                cur = pickle.load(open('medium-graph-IterFCS-size-'+str(size)+'-trial-'+str(trial)+'.pickle', 'rb'))
                lpAdvanced.append((cur['IterFCS'][0][-1]['Disparity'], cur['IterFCS'][0][-1]['Lift'], cur['IterFCS'][0][-1]['Time']))
                cur = pickle.load(
                    open('medium-graph-ForestFire-size-' + str(size) + '-trial-' + str(trial) + '.pickle', 'rb'))
                forestFire.append(
                    (cur['ForestFire'][-1]['Disparity'], cur['ForestFire'][-1]['Lift'], cur['ForestFire'][-1]['Time']))
                cur = pickle.load(open('medium-graph-CG-size-' + str(size) + '-trial-' + str(trial) + '.pickle', 'rb'))
                continuousGreedy.append((cur['CG']['Disparity'], cur['CG']['CS'], cur['CG']['Time']))

            cur = pickle.load(
                open('medium-graph-IRFA-size-' + str(size) + '-trial-' + str(trial) + '.pickle', 'rb'))
            irfa.append(
                ((cur['IRFA']['Disparity']-1)*100, (cur['IRFA']['Content Spread']-cur['Initial Content Spread'])/cur['Initial Content Spread']*100, cur['IRFA']['Time']))
            cur = pickle.load(
                open('medium-graph-SPGREEDY-size-' + str(size) + '-trial-' + str(trial) + '.pickle', 'rb'))
            spgreedy.append(
                ((cur['SPGREEDY']['Disparity']-1)*100, (cur['SPGREEDY']['Content Spread']-cur['Initial Content Spread'])/cur['Initial Content Spread']*100, cur['SPGREEDY']['Time']))
            cur = pickle.load(
                open('medium-graph-ACR-size-' + str(size) + '-trial-' + str(trial) + '.pickle', 'rb'))
            acr.append(
                ((cur['ACR']['Disparity']-1)*100, (cur['ACR']['Content Spread'] - cur['Initial Content Spread']) / cur[
                    'Initial Content Spread'] * 100, cur['ACR']['Time']))
            initialDisparity.append(cur['Initial Disparity'])


        row = []
        row.append(np.mean(initialDisparity))
        row.append(np.std(initialDisparity))
        for source in [lpAdvanced, continuousGreedy, forestFire, irfa, spgreedy, acr]:
            for i in range(3):
                row.append(np.mean([x[i] for x in source]))
                row.append(np.std([x[i] for x in source]))

        myWriter.writerow(row)

def CalculateVaryingSources(oneThousand=False):
    myFile = open('Varying-Sources-Results'+('-1000' if oneThousand else '')+'.csv', 'w')
    myWriter = csv.writer(myFile, delimiter=',')
    
    for k in [3,6,9,12]:
        contentSpread = []
        disparity = []
        time = []
        initialDisparity = []
        cgContentSpread = []
        cgDisparity = []
        cgTime = []
        irfaContentSpread = []
        irfaDisparity = []
        irfaTime = []
        spgreedyContentSpread = []
        spgreedyDisparity = []
        spgreedyTime = []
        acrContentSpread = []
        acrDisparity = []
        acrTime = []
        contentSpreadCommunity = []
        disparityCommunity = []
        timeCommunity = []
        cgContentSpreadCommunity = []
        cgDisparityCommunity = []
        cgTimeCommunity = []
        irfaContentSpreadCommunity = []
        irfaDisparityCommunity = []
        irfaTimeCommunity = []
        spgreedyContentSpreadCommunity = []
        spgreedyDisparityCommunity = []
        spgreedyTimeCommunity = []
        acrContentSpreadCommunity = []
        acrDisparityCommunity = []
        acrTimeCommunity = []
        i = 0
        j = 0
        for instance in range(20):
            fofResults = pickle.load(open('varying'+('-1000' if oneThousand else '')+'-sources-'+str(k)+'-trial-'+str(instance)+'-fof.pickle', 'rb'))
            communityResults = pickle.load(open('varying'+('-1000' if oneThousand else '')+'-sources-'+str(k)+'-trial-'+str(instance)+'-community.pickle', 'rb'))
            initialDisparity.append(fofResults['Initial Disparity'])

            if fofResults['Lift'] == 0:
                i = i + 1
                print(i)
            if communityResults['Lift'] == 0:
                j = j + 1
                print(j)

            contentSpread.append(fofResults['Lift'])
            disparity.append(fofResults['Final Disparity'])
            time.append(fofResults['Time'])
            cgContentSpread.append(fofResults['CG Lift'])
            cgDisparity.append(fofResults['CG Final Disparity'])
            cgTime.append(fofResults['CG Time'])
            irfaContentSpread.append(fofResults['IRFA Lift'])
            spgreedyContentSpread.append(fofResults['SPGREEDY Lift'])
            acrContentSpread.append(fofResults['ACR Lift'])
            irfaDisparity.append(fofResults['IRFA Final Disparity'])
            spgreedyDisparity.append(fofResults['SPGREEDY Final Disparity'])
            acrDisparity.append(fofResults['ACR Final Disparity'])
            irfaTime.append(fofResults['IRFA Time'])
            spgreedyTime.append(fofResults['SPGREEDY Time'])
            acrTime.append(fofResults['ACR Time'])

            contentSpreadCommunity.append(communityResults['Lift'])
            disparityCommunity.append(communityResults['Final Disparity'])
            timeCommunity.append(communityResults['Time'])
            cgContentSpreadCommunity.append(communityResults['CG Lift'])
            cgDisparityCommunity.append(communityResults['CG Final Disparity'])
            cgTimeCommunity.append(communityResults['CG Time'])
            irfaContentSpreadCommunity.append(communityResults['IRFA Lift'])
            spgreedyContentSpreadCommunity.append(communityResults['SPGREEDY Lift'])
            acrContentSpreadCommunity.append(communityResults['ACR Lift'])
            irfaDisparityCommunity.append(communityResults['IRFA Final Disparity'])
            spgreedyDisparityCommunity.append(communityResults['SPGREEDY Final Disparity'])
            acrDisparityCommunity.append(communityResults['ACR Final Disparity'])
            irfaTimeCommunity.append(communityResults['IRFA Time'])
            spgreedyTimeCommunity.append(communityResults['SPGREEDY Time'])
            acrTimeCommunity.append(communityResults['ACR Time'])

        averageInitialDisparity = np.mean(initialDisparity)
        initialDisparityStandardDeviation = np.std(initialDisparity)

        averageContentSpread = np.mean(contentSpread)
        contentSpreadStandardDeviation = np.std(contentSpread)
        averageDisparity = np.mean(disparity)
        disparityStandardDeviation = np.std(disparity)
        averageTime = np.mean(time)
        timeStandardDeviation = np.std(time)
        averageCGContentSpread = np.mean(cgContentSpread)
        cgContentSpreadStandardDeviation = np.std(cgContentSpread)
        averageCGDisparity = np.mean(cgDisparity)
        cgDisparityStandardDeviation = np.std(cgDisparity)
        averageCGTime = np.mean(cgTime)
        cgTimeStandardDeviation = np.std(cgTime)
        irfaAverageContentSpread = np.mean(irfaContentSpread)
        irfaContentSpreadStandardDeviation = np.std(irfaContentSpread)
        irfaAverageDisparity = np.mean(irfaDisparity)
        irfaDisparityStandardDeviation = np.std(irfaDisparity)
        irfaAverageTime = np.mean(irfaTime)
        irfaTimeStandardDeviation = np.std(irfaTime)
        spgreedyAverageContentSpread = np.mean(spgreedyContentSpread)
        spgreedyContentSpreadStandardDeviation = np.std(spgreedyContentSpread)
        spgreedyAverageDisparity = np.mean(spgreedyDisparity)
        spgreedyDisparityStandardDeviation = np.std(spgreedyDisparity)
        spgreedyAverageTime = np.mean(spgreedyTime)
        spgreedyTimeStandardDeviation = np.std(spgreedyTime)
        acrAverageContentSpread = np.mean(acrContentSpread)
        acrContentSpreadStandardDeviation = np.std(acrContentSpread)
        acrAverageDisparity = np.mean(acrDisparity)
        acrDisparityStandardDeviation = np.std(acrDisparity)
        acrAverageTime = np.mean(acrTime)
        acrTimeStandardDeviation = np.std(acrTime)

        averageContentSpreadCommunity = np.mean(contentSpreadCommunity)
        contentSpreadStandardDeviationCommunity = np.std(contentSpreadCommunity)
        averageDisparityCommunity = np.mean(disparityCommunity)
        disparityStandardDeviationCommunity = np.std(disparityCommunity)
        averageTimeCommunity = np.mean(timeCommunity)
        timeStandardDeviationCommunity = np.std(timeCommunity)
        averageCGContentSpreadCommunity = np.mean(cgContentSpreadCommunity)
        cgContentSpreadStandardDeviationCommunity = np.std(cgContentSpreadCommunity)
        averageCGDisparityCommunity = np.mean(cgDisparityCommunity)
        cgDisparityStandardDeviationCommunity = np.std(cgDisparityCommunity)
        averageCGTimeCommunity = np.mean(cgTimeCommunity)
        cgTimeStandardDeviationCommunity = np.std(cgTimeCommunity)
        irfaAverageContentSpreadCommunity = np.mean(irfaContentSpreadCommunity)
        irfaContentSpreadStandardDeviationCommunity = np.std(irfaContentSpreadCommunity)
        irfaAverageDisparityCommunity = np.mean(irfaDisparityCommunity)
        irfaDisparityStandardDeviationCommunity = np.std(irfaDisparityCommunity)
        irfaAverageTimeCommunity = np.mean(irfaTimeCommunity)
        irfaTimeStandardDeviationCommunity = np.std(irfaTimeCommunity)
        spgreedyAverageContentSpreadCommunity = np.mean(spgreedyContentSpreadCommunity)
        spgreedyContentSpreadStandardDeviationCommunity = np.std(spgreedyContentSpreadCommunity)
        spgreedyAverageDisparityCommunity = np.mean(spgreedyDisparityCommunity)
        spgreedyDisparityStandardDeviationCommunity = np.std(spgreedyDisparityCommunity)
        spgreedyAverageTimeCommunity = np.mean(spgreedyTimeCommunity)
        spgreedyTimeStandardDeviationCommunity = np.std(spgreedyTimeCommunity)
        acrAverageContentSpreadCommunity = np.mean(acrContentSpreadCommunity)
        acrContentSpreadStandardDeviationCommunity = np.std(acrContentSpreadCommunity)
        acrAverageDisparityCommunity = np.mean(acrDisparityCommunity)
        acrDisparityStandardDeviationCommunity = np.std(acrDisparityCommunity)
        acrAverageTimeCommunity = np.mean(acrTimeCommunity)
        acrTimeStandardDeviationCommunity = np.std(acrTimeCommunity)

        myWriter.writerow([k, averageInitialDisparity, initialDisparityStandardDeviation, averageContentSpread, contentSpreadStandardDeviation, averageDisparity, disparityStandardDeviation, averageTime, timeStandardDeviation, averageCGContentSpread, cgContentSpreadStandardDeviation, averageCGDisparity, cgDisparityStandardDeviation, averageCGTime, cgTimeStandardDeviation, irfaAverageContentSpread, irfaContentSpreadStandardDeviation, irfaAverageDisparity, irfaDisparityStandardDeviation, irfaAverageTime, irfaTimeStandardDeviation, spgreedyAverageContentSpread, spgreedyContentSpreadStandardDeviation, spgreedyAverageDisparity, spgreedyDisparityStandardDeviation, spgreedyAverageTime, spgreedyTimeStandardDeviation, acrAverageContentSpread, acrContentSpreadStandardDeviation, acrAverageDisparity, acrDisparityStandardDeviation, acrAverageTime, acrTimeStandardDeviation, averageContentSpreadCommunity, contentSpreadStandardDeviationCommunity, averageDisparityCommunity, disparityStandardDeviationCommunity, averageTimeCommunity, timeStandardDeviationCommunity, averageCGContentSpreadCommunity, cgContentSpreadStandardDeviationCommunity, averageCGDisparityCommunity, cgDisparityStandardDeviationCommunity, averageCGTimeCommunity, cgTimeStandardDeviationCommunity, irfaAverageContentSpreadCommunity, irfaContentSpreadStandardDeviationCommunity, irfaAverageDisparityCommunity, irfaDisparityStandardDeviationCommunity, irfaAverageTimeCommunity, irfaTimeStandardDeviationCommunity, spgreedyAverageContentSpreadCommunity, spgreedyContentSpreadStandardDeviationCommunity, spgreedyAverageDisparityCommunity, spgreedyDisparityStandardDeviationCommunity, spgreedyAverageTimeCommunity, spgreedyTimeStandardDeviationCommunity, acrAverageContentSpreadCommunity, acrContentSpreadStandardDeviationCommunity, acrAverageDisparityCommunity, acrDisparityStandardDeviationCommunity, acrAverageTimeCommunity, acrTimeStandardDeviationCommunity])
def CalculateVaryingSourcesMultiFile(oneThousand=False):
    liftFile = open('Varying-Sources-Results-Lift'+('-1000' if oneThousand else '')+'.csv', 'w')
    fairFile = open('Varying-Sources-Results-Fair'+('-1000' if oneThousand else '')+'.csv', 'w')
    timeFile = open('Varying-Sources-Results-Time' + ('-1000' if oneThousand else '') + '.csv', 'w')

    liftWriter = csv.writer(liftFile, delimiter=',')
    fairWriter = csv.writer(fairFile, delimiter=',')
    timeWriter = csv.writer(timeFile, delimiter=',')

    for k in [3,6,9,12]:
        contentSpread = []
        disparity = []
        time = []
        initialDisparity = []
        cgContentSpread = []
        cgDisparity = []
        cgTime = []
        irfaContentSpread = []
        irfaDisparity = []
        irfaTime = []
        spgreedyContentSpread = []
        spgreedyDisparity = []
        spgreedyTime = []
        acrContentSpread = []
        acrDisparity = []
        acrTime = []
        contentSpreadCommunity = []
        disparityCommunity = []
        timeCommunity = []
        cgContentSpreadCommunity = []
        cgDisparityCommunity = []
        cgTimeCommunity = []
        irfaContentSpreadCommunity = []
        irfaDisparityCommunity = []
        irfaTimeCommunity = []
        spgreedyContentSpreadCommunity = []
        spgreedyDisparityCommunity = []
        spgreedyTimeCommunity = []
        acrContentSpreadCommunity = []
        acrDisparityCommunity = []
        acrTimeCommunity = []
        i = 0
        j = 0
        for instance in range(20):
            fofResults = pickle.load(open('varying'+('-1000' if oneThousand else '')+'-sources-'+str(k)+'-trial-'+str(instance)+'-fof.pickle', 'rb'))
            communityResults = pickle.load(open('varying'+('-1000' if oneThousand else '')+'-sources-'+str(k)+'-trial-'+str(instance)+'-community.pickle', 'rb'))
            initialDisparity.append(fofResults['Initial Disparity'])

            if fofResults['Lift'] == 0:
                i = i + 1
                print(i)
            if communityResults['Lift'] == 0:
                j = j + 1
                print(j)

            contentSpread.append(fofResults['Lift'])
            disparity.append(fofResults['Final Disparity'])
            time.append(fofResults['Time'])
            cgContentSpread.append(fofResults['CG Lift'])
            cgDisparity.append(fofResults['CG Final Disparity'])
            cgTime.append(fofResults['CG Time'])
            irfaContentSpread.append(fofResults['IRFA Lift'])
            spgreedyContentSpread.append(fofResults['SPGREEDY Lift'])
            acrContentSpread.append(fofResults['ACR Lift'])
            irfaDisparity.append(fofResults['IRFA Final Disparity'])
            spgreedyDisparity.append(fofResults['SPGREEDY Final Disparity'])
            acrDisparity.append(fofResults['ACR Final Disparity'])
            irfaTime.append(fofResults['IRFA Time'])
            spgreedyTime.append(fofResults['SPGREEDY Time'])
            acrTime.append(fofResults['ACR Time'])

            contentSpreadCommunity.append(communityResults['Lift'])
            disparityCommunity.append(communityResults['Final Disparity'])
            timeCommunity.append(communityResults['Time'])
            cgContentSpreadCommunity.append(communityResults['CG Lift'])
            cgDisparityCommunity.append(communityResults['CG Final Disparity'])
            cgTimeCommunity.append(communityResults['CG Time'])
            irfaContentSpreadCommunity.append(communityResults['IRFA Lift'])
            spgreedyContentSpreadCommunity.append(communityResults['SPGREEDY Lift'])
            acrContentSpreadCommunity.append(communityResults['ACR Lift'])
            irfaDisparityCommunity.append(communityResults['IRFA Final Disparity'])
            spgreedyDisparityCommunity.append(communityResults['SPGREEDY Final Disparity'])
            acrDisparityCommunity.append(communityResults['ACR Final Disparity'])
            irfaTimeCommunity.append(communityResults['IRFA Time'])
            spgreedyTimeCommunity.append(communityResults['SPGREEDY Time'])
            acrTimeCommunity.append(communityResults['ACR Time'])

        averageInitialDisparity = np.mean(initialDisparity)
        initialDisparityStandardDeviation = np.std(initialDisparity)

        averageContentSpread = np.mean(contentSpread)
        contentSpreadStandardDeviation = np.std(contentSpread)
        averageDisparity = np.mean(disparity)
        disparityStandardDeviation = np.std(disparity)
        averageTime = np.mean(time)
        timeStandardDeviation = np.std(time)
        averageCGContentSpread = np.mean(cgContentSpread)
        cgContentSpreadStandardDeviation = np.std(cgContentSpread)
        averageCGDisparity = np.mean(cgDisparity)
        cgDisparityStandardDeviation = np.std(cgDisparity)
        averageCGTime = np.mean(cgTime)
        cgTimeStandardDeviation = np.std(cgTime)
        irfaAverageContentSpread = np.mean(irfaContentSpread)
        irfaContentSpreadStandardDeviation = np.std(irfaContentSpread)
        irfaAverageDisparity = np.mean(irfaDisparity)
        irfaDisparityStandardDeviation = np.std(irfaDisparity)
        irfaAverageTime = np.mean(irfaTime)
        irfaTimeStandardDeviation = np.std(irfaTime)
        spgreedyAverageContentSpread = np.mean(spgreedyContentSpread)
        spgreedyContentSpreadStandardDeviation = np.std(spgreedyContentSpread)
        spgreedyAverageDisparity = np.mean(spgreedyDisparity)
        spgreedyDisparityStandardDeviation = np.std(spgreedyDisparity)
        spgreedyAverageTime = np.mean(spgreedyTime)
        spgreedyTimeStandardDeviation = np.std(spgreedyTime)
        acrAverageContentSpread = np.mean(acrContentSpread)
        acrContentSpreadStandardDeviation = np.std(acrContentSpread)
        acrAverageDisparity = np.mean(acrDisparity)
        acrDisparityStandardDeviation = np.std(acrDisparity)
        acrAverageTime = np.mean(acrTime)
        acrTimeStandardDeviation = np.std(acrTime)

        averageContentSpreadCommunity = np.mean(contentSpreadCommunity)
        contentSpreadStandardDeviationCommunity = np.std(contentSpreadCommunity)
        averageDisparityCommunity = np.mean(disparityCommunity)
        disparityStandardDeviationCommunity = np.std(disparityCommunity)
        averageTimeCommunity = np.mean(timeCommunity)
        timeStandardDeviationCommunity = np.std(timeCommunity)
        averageCGContentSpreadCommunity = np.mean(cgContentSpreadCommunity)
        cgContentSpreadStandardDeviationCommunity = np.std(cgContentSpreadCommunity)
        averageCGDisparityCommunity = np.mean(cgDisparityCommunity)
        cgDisparityStandardDeviationCommunity = np.std(cgDisparityCommunity)
        averageCGTimeCommunity = np.mean(cgTimeCommunity)
        cgTimeStandardDeviationCommunity = np.std(cgTimeCommunity)
        irfaAverageContentSpreadCommunity = np.mean(irfaContentSpreadCommunity)
        irfaContentSpreadStandardDeviationCommunity = np.std(irfaContentSpreadCommunity)
        irfaAverageDisparityCommunity = np.mean(irfaDisparityCommunity)
        irfaDisparityStandardDeviationCommunity = np.std(irfaDisparityCommunity)
        irfaAverageTimeCommunity = np.mean(irfaTimeCommunity)
        irfaTimeStandardDeviationCommunity = np.std(irfaTimeCommunity)
        spgreedyAverageContentSpreadCommunity = np.mean(spgreedyContentSpreadCommunity)
        spgreedyContentSpreadStandardDeviationCommunity = np.std(spgreedyContentSpreadCommunity)
        spgreedyAverageDisparityCommunity = np.mean(spgreedyDisparityCommunity)
        spgreedyDisparityStandardDeviationCommunity = np.std(spgreedyDisparityCommunity)
        spgreedyAverageTimeCommunity = np.mean(spgreedyTimeCommunity)
        spgreedyTimeStandardDeviationCommunity = np.std(spgreedyTimeCommunity)
        acrAverageContentSpreadCommunity = np.mean(acrContentSpreadCommunity)
        acrContentSpreadStandardDeviationCommunity = np.std(acrContentSpreadCommunity)
        acrAverageDisparityCommunity = np.mean(acrDisparityCommunity)
        acrDisparityStandardDeviationCommunity = np.std(acrDisparityCommunity)
        acrAverageTimeCommunity = np.mean(acrTimeCommunity)
        acrTimeStandardDeviationCommunity = np.std(acrTimeCommunity)


        liftWriter.writerow([k, round(averageContentSpread, 2), round(contentSpreadStandardDeviation, 2),  round(averageCGContentSpread, 2), round(cgContentSpreadStandardDeviation, 2),
                           round(irfaAverageContentSpread, 2), round(irfaContentSpreadStandardDeviation, 2), round(spgreedyAverageContentSpread, 2), round(spgreedyContentSpreadStandardDeviation, 2),
                           round(acrAverageContentSpread, 2), round(acrContentSpreadStandardDeviation, 2)])
        liftWriter.writerow([k, round(averageContentSpreadCommunity, 2), round(contentSpreadStandardDeviationCommunity, 2),
                           round(averageCGContentSpreadCommunity, 2), round(cgContentSpreadStandardDeviationCommunity, 2), round(irfaAverageContentSpreadCommunity, 2),
                            round(irfaContentSpreadStandardDeviationCommunity, 2), round(spgreedyAverageContentSpreadCommunity, 2),
                           round(spgreedyContentSpreadStandardDeviationCommunity, 2), round(acrAverageContentSpreadCommunity, 2),
                           round(acrContentSpreadStandardDeviationCommunity, 2)])

        fairWriter.writerow([k, round(averageInitialDisparity, 2), round(initialDisparityStandardDeviation, 2), round(averageDisparity, 2), round(disparityStandardDeviation, 2),
                           round(averageCGDisparity, 2), round(cgDisparityStandardDeviation, 2),  round(irfaAverageDisparity, 2),
                           round(irfaDisparityStandardDeviation, 2),round(spgreedyAverageDisparity, 2), round(spgreedyDisparityStandardDeviation, 2),
                           round(acrAverageDisparity, 2), round(acrDisparityStandardDeviation, 2)])
        fairWriter.writerow([k, round(averageInitialDisparity, 2), round(initialDisparityStandardDeviation, 2),
                           round(averageDisparityCommunity, 2), round(disparityStandardDeviationCommunity, 2), round(averageCGDisparityCommunity, 2),
                           round(cgDisparityStandardDeviationCommunity, 2), round(irfaAverageDisparityCommunity, 2),
                           round(irfaDisparityStandardDeviationCommunity, 2), round(spgreedyAverageDisparityCommunity, 2),
                           round(spgreedyDisparityStandardDeviationCommunity, 2), round(acrAverageDisparityCommunity, 2),
                           round(acrDisparityStandardDeviationCommunity, 2)])

        timeWriter.writerow([k, round(averageTime, 2), round(timeStandardDeviation, 2), round(averageCGTime, 2), round(cgTimeStandardDeviation, 2),
                            round(irfaAverageTime, 2), round(irfaTimeStandardDeviation, 2), round(spgreedyAverageTime, 2), round(spgreedyTimeStandardDeviation, 2),
                             round(acrAverageTime, 2), round(acrTimeStandardDeviation, 2)])
        timeWriter.writerow([k, round(averageTimeCommunity, 2), round(timeStandardDeviationCommunity, 2),
                           round(averageCGTimeCommunity, 2), round(cgTimeStandardDeviationCommunity, 2), round(irfaAverageTimeCommunity, 2),
                           round(irfaTimeStandardDeviationCommunity, 2), round(spgreedyAverageTimeCommunity, 2),
                           round(spgreedyTimeStandardDeviationCommunity, 2), round(acrAverageTimeCommunity, 2),
                           round(acrTimeStandardDeviationCommunity, 2)])

def CalculateVaryingSourcesFairness():
    myFile = open('Varying-Sources-Fairness-Results-Lift.csv', 'w')
    myWriter = csv.writer(myFile, delimiter=',')
    myFile2 = open('Varying-Sources-Fairness-Results-Disparity.csv', 'w')
    myWriter2 = csv.writer(myFile2, delimiter=',')

    for k in [3, 6, 9, 12]:
        contentSpread = []
        unfairContentSpread = []
        disparity = []
        unfairDisparity = []
        initialDisparity = []
        for instance in range(20):
            results = pickle.load(open('varying-sources-' + str(k) + '-trial-' + str(instance) + '.pickle', 'rb'))
            contentSpread.append(results['Lift'])
            unfairContentSpread.append(results['UF-Lift'])
            disparity.append(results['Final Disparity'])
            unfairDisparity.append(results['UF-Final Disparity'])
            initialDisparity.append(results['Initial Disparity'])

        averageContentSpread = np.mean(contentSpread)
        contentSpreadStandardDeviation = np.std(contentSpread)
        averageUnfairContentSpread = np.mean(unfairContentSpread)
        unfairContentSpreadStandardDeviation = np.std(unfairContentSpread)
        averageDisparity = np.mean(disparity)
        disparityStandardDeviation = np.std(disparity)
        averageUnfairDisparity = np.mean(unfairDisparity)
        unfairDisparityStandardDeviation = np.std(unfairDisparity)
        averageInitialDisparity = np.mean(initialDisparity)
        initialDisparityStandardDeviation = np.std(initialDisparity)

        myWriter.writerow([k, averageContentSpread, contentSpreadStandardDeviation,
                           averageUnfairContentSpread, unfairContentSpreadStandardDeviation])
        myWriter2.writerow([k, averageInitialDisparity, initialDisparityStandardDeviation, averageDisparity, disparityStandardDeviation,
                           averageUnfairDisparity, unfairDisparityStandardDeviation])



def CalculateOptimality():
    myFile = open('Optimality-Results.csv', 'w')
    myWriter = csv.writer(myFile, delimiter=',')
    
    points = []

    for experiment in [11, 13, 14, 19, 3, 6, 20]:
        print(experiment)
        points.append(pickle.load(open('Optimality-Experiment-Updated-'+str(experiment)+'.pickle', 'rb')))

    points.sort(key=lambda e: e['Initial Disparity'])

    for point in points:
        for x in point['IterFCS']:
            print(x)
        myWriter.writerow((point['Initial Disparity'], point['Optimum']['Disparity'], point['Optimum']['Content Spread'], np.mean([x[0]['Disparity'] for x in point['IterFCS']]), np.mean([x[0]['Lift'] for x in point['IterFCS']]), np.std([x[0]['Disparity'] for x in point['IterFCS']]), np.std([x[0]['Lift'] for x in point['IterFCS']])))

def CalculateScaling():
    myFile = open('Scaling-Results.csv', 'w')
    myWriter = csv.writer(myFile, delimiter=',')

    for i in [50000, 100000, 200000, 500000]:
        results = pickle.load(open('forest-fire-scale-results4-'+str(i)+'.pickle', 'rb'))
        myWriter.writerow((results['Initial Disparity'], results['ForestFire'][-1]['Disparity'], results['ForestFire'][-1]['Lift'], results['ForestFire'][-1]['Time']))

#CalculateOptimality()
# CalculateVaryingKValues()
# CalculateVaryingSources()
# CalculateVaryingPValues()
# CalculateVaryingKValues(True)
# CalculateVaryingSources(True)
# CalculateVaryingPValues(True)
#CalculateVaryingKValuesMultiFile()
#CalculateVaryingSourcesMultiFile()
#CalculateVaryingPValuesMultiFile()
#CalculateVaryingKValuesMultiFile(True)
#CalculateVaryingSourcesMultiFile(True)
#CalculateVaryingPValuesMultiFile(True)
# CalculateAlgorithmsComparison2()
#CalculateForestFireValues()
#CalculateScaling()
CalculateVaryingSourcesFairness()
