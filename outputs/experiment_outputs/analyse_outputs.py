import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import matplotlib as mpl


def resolve_micro_metrics_path(output_dir, split, filename=''):
    """
    Resolve path to micro metric files written by ood_evaluation().

    Matches Micro_metrics_{ID|OOD}<suffix>.txt from evaluate_network_utils.py,
    with a fallback for lowercase micro_metrics_* filenames.
    """
    suffix = filename if (filename.startswith('_') or filename == '') else f'_{filename}'
    candidates = [
        os.path.join(output_dir, f'Micro_metrics_{split}{suffix}.txt'),
        os.path.join(output_dir, f'micro_metrics_{split}{suffix}.txt'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f'Could not find {split} micro metrics file in {output_dir}. '
        f'Tried: {[os.path.basename(p) for p in candidates]}. '
        f'Run evaluate_OOD_detection_method.py with --save_results_micro True and matching --filename.'
    )


class Visualise_outputs():

    def __init__(self, filename='', output_dir=None):
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = output_dir
        self.filename = filename

        id_path = resolve_micro_metrics_path(output_dir, 'ID', filename)
        ood_path = resolve_micro_metrics_path(output_dir, 'OOD', filename)
        self.df_ID = pd.read_csv(id_path, on_bad_lines='skip')
        self.df_OOD = pd.read_csv(ood_path, on_bad_lines='skip')

        self.OOD_detection_methods = self.df_ID['OOD detection method'].unique()
        self.unique_ID_images = self.df_ID['Image'].unique()
        self.unique_OOD_images = self.df_OOD['Image'].unique()
        self.classes = self.df_ID['Target'].unique()


    def get_accuracy_AUC(self,method_name='all'):

        auroc_accuracy_ID = []
        auroc_accuracy_OOD = []
        auroc_accuracy_weighted = []
        auroc_accuracy = []

        aucpr_accuracy_ID = []
        aucpr_accuracy_OOD = []
        aucpr_accuracy_weighted = []
        aucpr_accuracy = []

        method_names = []
        
        for method in self.OOD_detection_methods:
            if method_name != 'all' and method_name != method:
                pass
            else:
                df_ID_method = self.df_ID[self.df_ID['OOD detection method']==method]
                df_OOD_method = self.df_OOD[self.df_OOD['OOD detection method']==method]
                
                ID_correct = df_ID_method['Correct']
                OOD_correct = df_OOD_method['Correct']

                ID_metric = df_ID_method['Metric']
                OOD_metric = df_OOD_method['Metric']

                Correct = list(ID_correct) + list(OOD_correct)
                #input(sum(ID_correct))
                #input(sum(OOD_correct))
                Metric = list(ID_metric) + list(OOD_metric)

                weight_ID = np.full(len(ID_metric),1/len(ID_metric))
                weight_OOD = np.full(len(OOD_metric),1/len(OOD_metric))
                sample_weight = list(weight_ID) + list(weight_OOD)

                auroc_ID,aucpr_ID = self.get_auroc_aucpr(ID_correct, ID_metric)
                auroc_OOD,aucpr_OOD = self.get_auroc_aucpr(OOD_correct, OOD_metric)
                auroc_weighted,aucpr_weighted = self.get_auroc_aucpr(Correct, Metric,sample_weight=sample_weight)
                auroc,aucpr = self.get_auroc_aucpr(Correct, Metric)

                auroc_accuracy_ID.append(auroc_ID)
                auroc_accuracy_OOD.append(auroc_OOD)
                auroc_accuracy_weighted.append(auroc_weighted)
                auroc_accuracy.append(auroc)

                aucpr_accuracy_ID.append(aucpr_ID)
                aucpr_accuracy_OOD.append(aucpr_OOD)
                aucpr_accuracy_weighted.append(aucpr_weighted)
                aucpr_accuracy.append(aucpr)

                method_names.append(method)

        return auroc_accuracy_ID, auroc_accuracy_OOD, auroc_accuracy_weighted, auroc_accuracy, aucpr_accuracy_ID, aucpr_accuracy_OOD, aucpr_accuracy_weighted, aucpr_accuracy, method_names
    
    def plot_accuracy_AUC(self,method_name='all'):
        """
        Plot metrics related to accuracy.

        Parameters
        ----------
        method_name : str, optional
            The name of the OOD detection method to plot. The default is 'all'.
        """
        auroc_accuracy_ID, auroc_accuracy_OOD, auroc_accuracy_weighted, auroc_accuracy, aucpr_accuracy_ID, aucpr_accuracy_OOD, aucpr_accuracy_weighted, aucpr_accuracy, method_names = self.get_accuracy_AUC(method_name)

        self.set_style()
        if len(auroc_accuracy_ID) >= 6:
            plt.title('Accuracy AUROC')
            plt.plot(auroc_accuracy_ID,color='RoyalBlue',label='ID')
            plt.plot(auroc_accuracy_OOD,color='Orange',label='OOD')
            plt.plot(auroc_accuracy_weighted,color=(160/256, 135/256, 113/256),label='Bal ID OOD')
            plt.plot(auroc_accuracy,color='lightgray',label='All data')
            plt.legend()
            plt.xlabel('Metric')
            plt.ylabel('AUROC')
            plt.xlim(0,len(auroc_accuracy_ID))
            plt.grid(alpha=0.3)
            plt.show()
        else:
            self.plot_grouped_bar([['ID','RoyalBlue',auroc_accuracy_ID],['OOD','Orange',auroc_accuracy_OOD],['Bal ID OOD',(160/256, 135/256, 113/256),auroc_accuracy_weighted],['All data','lightgray',auroc_accuracy]],method_names,ylabel='AUROC')
 
        if len(auroc_accuracy_ID) >= 6:
            plt.title('Accuracy AUCPR')
            plt.plot(aucpr_accuracy_ID,color='RoyalBlue',label='ID')
            plt.plot(aucpr_accuracy_OOD,color='Orange',label='OOD')
            plt.plot(aucpr_accuracy_weighted,color=(160/256, 135/256, 113/256),label='Bal ID OOD')
            plt.plot(aucpr_accuracy,color='lightgray',label='All data')
            plt.legend()
            plt.xlabel('Accuracy Metric')
            plt.ylabel('AUCPR')
            plt.xlim(0,len(auroc_accuracy_ID))
            plt.grid(alpha=0.3)
            plt.show()
        else:
            self.plot_grouped_bar([['ID','RoyalBlue',aucpr_accuracy_ID],['OOD','Orange',aucpr_accuracy_OOD],['Bal ID OOD',(160/256, 135/256, 113/256),aucpr_accuracy_weighted],['All data','lightgray',aucpr_accuracy]],method_names,ylabel='AUCPR')


    def get_OOD_AUC(self,method_name='all'):
        """
        Function calculates the AUROC and AUCPR for OOD detection.

        Parameters
        -----------
        method_name : str, optional
            The name of the OOD detection method to plot. The default is 'all'.
        """
        AUROC_OOD = []
        AUCPR_OOD = []
        method_names = []
        
        for method in self.OOD_detection_methods:
            if method_name != 'all' and method_name != method:
                pass
            else:
                df_ID_method = self.df_ID[self.df_ID['OOD detection method']==method]
                df_OOD_method = self.df_OOD[self.df_OOD['OOD detection method']==method]

                ID_correct = np.ones(len(df_ID_method))
                OOD_correct = np.zeros(len(df_OOD_method))

                ID_metric = df_ID_method['Metric']
                OOD_metric = df_OOD_method['Metric']

                Correct = list(ID_correct) + list(OOD_correct)
                Metric = list(ID_metric) + list(OOD_metric)

                auroc,aucpr = self.get_auroc_aucpr(Correct, Metric)
                AUROC_OOD.append(auroc)
                AUCPR_OOD.append(aucpr)

                method_names.append(method)
            
        return AUROC_OOD,AUCPR_OOD,method_names
    

    def plot_OOD_AUC(self,method_name='all'):
        """
        Plot metrics related to OOD detection.

        Parameters
        ----------
        method_name : str, optional
            The name of the OOD detection method to plot. The default is 'all'.
        """
        AUROC_OOD,AUCPR_OOD,method_names = self.get_OOD_AUC(method_name)

        self.set_style()
        if len(AUROC_OOD) >= 6:
            plt.title('OOD detection AUROC')
            plt.plot(AUROC_OOD,color='crimson')
            plt.xlabel('Metric')
            plt.ylabel('OOD detection AUROC')
            plt.xlim(0,len(AUROC_OOD))
            plt.grid(alpha=0.3)
            plt.show()
        else:
            self.plot_grouped_bar([['AUROC','crimson',AUROC_OOD]],method_names,ylabel='AUROC')

        self.set_style()
        if len(AUCPR_OOD) >= 6:
            plt.title('OOD detection AUCPR')
            plt.plot(AUCPR_OOD,color='crimson')
            plt.xlabel('Metric')
            plt.ylabel('OOD detection AUCPR')
            plt.xlim(0,len(AUCPR_OOD))
            plt.grid(alpha=0.3)
            plt.show()
        else:
            self.plot_grouped_bar([['AUCPR','crimson',AUCPR_OOD]],method_names,ylabel='AUCPR')

        
    def plot_OOD_accuracy_AUC(self,method_name='all'):
        """
        Plot metrics related to OOD detection and accuracy.

        Parameters
        ----------
        method_name : str, optional
            The name of the OOD detection method to plot. The default is 'all'.
        """
        AUROC_OOD,AUCPR_OOD,method_names = self.get_OOD_AUC(method_name)
        auroc_accuracy_ID, auroc_accuracy_OOD, auroc_accuracy_weighted, auroc_accuracy, aucpr_accuracy_ID, aucpr_accuracy_OOD, aucpr_accuracy_weighted, aucpr_accuracy, method_names = self.get_accuracy_AUC(method_name)

        plt.plot(AUROC_OOD,label='OOD detection AUROC',color='crimson')
        plt.plot(auroc_accuracy,label='Accuracy AUROC',color='RoyalBlue')
        plt.grid(alpha=0.3)
        plt.xlim(0,len(AUROC_OOD))
        plt.ylabel('AUROC')
        plt.xlabel('Mahalanobis Module')
        plt.legend()
        plt.show()

    
    def get_auroc_aucpr(self,true_labels,pred_probs,sample_weight=None):
        """
        Function calculates the AUROC and AUCPR

        Parameters
        -----------
        true_labels : array
            True labels
        pred_probs : array
            Predicted probabilities
        sample_weight : array
            Sample weights

        Returns
        --------
        auroc : float
            AUROC
        aucpr : float
            AUCPR
        """
        pred_probs = (pred_probs - np.min(pred_probs))/(np.max(pred_probs) - np.min(pred_probs))
        fpr, tpr, _ = skm.roc_curve(y_true = true_labels, y_score = pred_probs, pos_label = 1,sample_weight=sample_weight) #positive class is 1; negative class is 0
        auroc = skm.auc(fpr, tpr)
        precision, recall, _ = skm.precision_recall_curve(true_labels, pred_probs)
        aucpr = skm.auc(recall, precision)

        return auroc,aucpr
    
    def calculate_aurc(self,true_labels, predictions, confidence):
        # Calculate residuals as binary incorrect/correct flags
        residuals = np.array([1 if p != t else 0 for p, t in zip(predictions, true_labels)])
        #residuals = np.array([0 if p != t else 1 for p, t in zip(predictions, true_labels)])

        n = len(residuals)
        idx_sorted = np.argsort(np.array(confidence))  # Sort indices by descending confidence
        #input(confidence[idx_sorted[-1]])
        #input(residuals[idx_sorted[-1]])
        coverages = []
        risks = []
        weights = []

        cov = n
        error_sum = sum(residuals[idx_sorted])
        coverages.append(cov / n)
        risks.append(error_sum / n)

        tmp_weight = 0

        for i in range(n - 1):
            cov -= 1
            error_sum -= residuals[idx_sorted[i]]
            selective_risk = error_sum / (n - 1 - i)
            tmp_weight += 1

            if i == 0 or confidence[idx_sorted[i]] != confidence[idx_sorted[i - 1]]:
                coverages.append(cov / n)
                risks.append(selective_risk)
                weights.append(tmp_weight / n)
                tmp_weight = 0

        if tmp_weight > 0:
            coverages.append(0)
            risks.append(risks[-1])
            weights.append(tmp_weight / n)

        self.set_style(fontsize=15)
        plt.plot(coverages,risks,color='seagreen')
        plt.xlabel('Coverage (%)')
        plt.ylabel('Risk')
        plt.xlim(0,1)
        plt.ylim(0)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        #print(risks)
        #print(coverages)
        plt.show()

        # Calculate AURC using a trapezoidal approximation
        aurc = sum([(risks[i] + risks[i+1]) * 0.5 * weights[i] for i in range(len(weights)-1)])
        return coverages, risks, aurc
    
    def plot_accuracy_AURC(self,method_name='all'):
        """
        Plot metrics related to OOD detection.

        Parameters
        ----------
        method_name : str, optional
            The name of the OOD detection method to plot. The default is 'all'.
        """
        AURC_ID, AURC_OOD, AURC, method_names = self.get_a_AURC(method_name)

        self.set_style()
        if len(AURC_OOD) >= 6:
            plt.title('OOD detection AUROC')
            plt.plot(AURC_OOD,color='crimson')
            plt.xlabel('Metric')
            plt.ylabel('OOD detection AUROC')
            plt.xlim(0,len(AURC_OOD))
            plt.grid(alpha=0.3)
            plt.show()
        else:
            self.plot_grouped_bar([['AURC','crimson',AURC_ID],['AURC','blue',AURC_OOD],['AURC','green',AURC]],method_names,ylabel='AUROC')


    def get_a_AURC(self,method_name='all'):
            aurc_accuracy_ID = []
            aurc_accuracy_OOD = []
            aurc_accuracy = []

            method_names = []
            
            for method in self.OOD_detection_methods:
                if method_name != 'all' and method_name != method:
                    pass
                else:
                    df_ID_method = self.df_ID[self.df_ID['OOD detection method']==method]
                    df_OOD_method = self.df_OOD[self.df_OOD['OOD detection method']==method]

                    ID_target = df_ID_method['Target']
                    OOD_target = df_OOD_method['Target']
                    ID_prediction = df_ID_method['Predicted']
                    OOD_prediction = df_OOD_method['Predicted']

                    #input(df_OOD_method['Predicted'])

                    ID_metric = df_ID_method['Metric']
                    OOD_metric = df_OOD_method['Metric']

                    Target = list(ID_target) + list(OOD_target)
                    Prediction = list(ID_prediction) + list(OOD_prediction)
                    Metric = list(ID_metric) + list(OOD_metric)

                    coverages, risks, aurc = self.calculate_aurc(Target, Prediction, Metric)
                    coverages_ID, risks_ID, aurc_ID = self.calculate_aurc(ID_target, ID_prediction, ID_metric)
                    coverages_OOD, risks_OOD, aurc_OOD = self.calculate_aurc(OOD_target, OOD_prediction, OOD_metric)

                    aurc_accuracy_ID.append(aurc_ID)
                    aurc_accuracy_OOD.append(aurc_OOD)
                    aurc_accuracy.append(aurc)


                    method_names.append(method)

            return aurc_accuracy_ID, aurc_accuracy_OOD, aurc_accuracy, method_names
        


    def get_accuracy_AURC(self,method_name='all'):
        """
        Function calculates the AUROC and AUCPR for OOD detection.

        Parameters
        -----------
        method_name : str, optional
            The name of the OOD detection method to plot. The default is 'all'.
        """
        AUROC_OOD = []
        AUCPR_OOD = []
        method_names = []
        
        for method in self.OOD_detection_methods:
            if method_name != 'all' and method_name != method:
                pass
            else:
                df_ID_method = self.df_ID[self.df_ID['OOD detection method']==method]
                df_OOD_method = self.df_OOD[self.df_OOD['OOD detection method']==method]

                ID_correct = np.ones(len(df_ID_method))
                OOD_correct = np.zeros(len(df_OOD_method))

                ID_metric = df_ID_method['Metric']
                OOD_metric = df_OOD_method['Metric']

                Correct = list(ID_correct) + list(OOD_correct)
                Metric = list(ID_metric) + list(OOD_metric)

                auroc,aucpr = self.get_auroc_aucpr(Correct, Metric)
                AUROC_OOD.append(auroc)
                AUCPR_OOD.append(aucpr)

                method_names.append(method)
            
        return AUROC_OOD,AUCPR_OOD,method_names


    def plot_hist_dist(self,x,bins=51,color='RoyalBlue',alpha=0.5,title=None,normalised=False,range=None,label=None,hatch=None):
        """
        Function plots a histogram of the data.

        Parameters
        -----------
        x : array
            Data to plot
        bins : int, optional
            Number of bins. The default is 51.
        color : str, optional
            Color of the histogram. The default is 'RoyalBlue'.
        alpha : float, optional
            Transparency of the histogram. The default is 0.5.
        title : str, optional
            Title of the plot. The default is None.
        normalised : bool, optional
            Whether to normalise the histogram. The default is False.
        range : tuple, optional
            Range of the histogram. The default is None.
        label : str, optional
            Label of the histogram. The default is None.
        hatch : str, optional
            Hatch pattern of the histogram. The default is None.
        """
        if range != None:
            vals,bins = np.histogram(x,bins = bins,range=range,density=normalised)
        else:
            vals,bins = np.histogram(x,bins = bins,density=normalised)
        bin_centers = (bins[1:]+bins[:-1])/2.0
        if title != None:
            plt.title(str(title),fontsize=18)
        plt.plot(bin_centers,vals,linewidth=2,color=color,marker="",label=label)
        plt.fill_between(bin_centers,vals,[0]*len(vals),color=color,alpha=alpha)
        if hatch != None:
            plt.fill_between(bin_centers,vals,[0]*len(vals),color='None',edgecolor=color,alpha=1,hatch=hatch)
        plt.ylim(0)
        plt.grid(alpha=0.3)


    def plot_grouped_bar(self,grouped_array,method_names,ylabel):
        """
        Function plots a grouped bar chart.

        Parameters
        -----------
        grouped_array : array
            Array of arrays containing the data to plot
        method_names : array
            Array of method names
        ylabel : str
            Label of the y-axis
        """
        if len(grouped_array[0][2]) > 2:
            self.set_style(fontsize=11)
        else:
            self.set_style(fontsize=15)
        fig, ax = plt.subplots(layout='constrained')
        x = np.arange(len(grouped_array[0][2]))

        width = 0.15  # the width of the bars
        multiplier = 0
        for name,color,arr in grouped_array:

            offset = width * multiplier
            rects = ax.bar(x + offset, arr, width,color=color,label=name)
            ax.bar_label(rects, fmt='%.3f',padding=3)
            multiplier += 1

        plt.grid(alpha=0.3)
        plt.legend()
        plt.xticks(x+width*1.5,method_names,rotation=20)
        plt.ylim(0,1)
        plt.ylabel(ylabel)
        plt.xlim(-width,max(x)+width*6.1)
        plt.show()




    def confusion_matrix(self, method_name=None):
        """
        Function plots the confusion matrix for the ID and OOD data.
        """
        if method_name is None:
            method_name = self.OOD_detection_methods[0]
        method = method_name

        df_ID_method = self.df_ID[self.df_ID['OOD detection method'] == method]
        df_OOD_method = self.df_OOD[self.df_OOD['OOD detection method'] == method]

        df_ID_method_pred = df_ID_method['Predicted']
        df_OOD_method_pred = df_OOD_method['Predicted']
        df_ID_method_target = df_ID_method['Target']
        df_OOD_method_target = df_OOD_method['Target']

        self.set_style(fontsize=12)

        plt.figure(figsize=(12, 12))

        classes = np.sort(self.classes)

        # Compute accuracies
        ID_correct = [1 if x == y else 0 for x, y in zip(df_ID_method_pred, df_ID_method_target)]
        ID_accuracy = sum(ID_correct) / len(ID_correct) if len(ID_correct) > 0 else 0
        OOD_correct = [1 if x == y else 0 for x, y in zip(df_OOD_method_pred, df_OOD_method_target)]
        OOD_accuracy = sum(OOD_correct) / len(OOD_correct) if len(OOD_correct) > 0 else 0


        ID_bal_accuracy = skm.balanced_accuracy_score(df_ID_method_target, df_ID_method_pred)
        OOD_bal_accuracy = skm.balanced_accuracy_score(df_OOD_method_target, df_OOD_method_pred)

        # Ensure all classes are represented in the confusion matrix
        labels = classes

        # Compute confusion matrices
        cm_ID = skm.confusion_matrix(df_ID_method_target, df_ID_method_pred, labels=labels)
        cm_ID_normalized = skm.confusion_matrix(df_ID_method_target, df_ID_method_pred, labels=labels, normalize='true')
        cm_OOD = skm.confusion_matrix(df_OOD_method_target, df_OOD_method_pred, labels=labels)
        cm_OOD_normalized = skm.confusion_matrix(df_OOD_method_target, df_OOD_method_pred, labels=labels, normalize='true')

        max_value = max(np.max(cm_ID_normalized), np.max(cm_OOD_normalized))
        min_value = min(np.min(cm_ID_normalized), np.min(cm_OOD_normalized))

        # Plotting
        ax = plt.subplot(2, 2, 1)
        plt.title(f'ID test data, acc. {ID_accuracy:.3f}, bal. acc. {ID_bal_accuracy:.3f}', color='RoyalBlue')
        disp = skm.ConfusionMatrixDisplay(confusion_matrix=cm_ID, display_labels=classes)
        disp.plot(ax=ax, cmap='binary')

        ax = plt.subplot(2, 2, 2)
        plt.title('ID test data (%)', color='RoyalBlue')
        disp = skm.ConfusionMatrixDisplay(confusion_matrix=cm_ID_normalized, display_labels=classes)
        disp.plot(ax=ax, cmap='binary', values_format=".2f")
        plt.imshow(cm_ID_normalized, interpolation='nearest', cmap='binary', vmin=min_value, vmax=max_value)

        ax = plt.subplot(2, 2, 3)
        plt.title(f'OOD test data, acc. {OOD_accuracy:.3f}, bal. acc. {OOD_bal_accuracy:.3f}', color='RoyalBlue')
        disp = skm.ConfusionMatrixDisplay(confusion_matrix=cm_OOD, display_labels=classes)
        disp.plot(ax=ax, cmap='binary')

        ax = plt.subplot(2, 2, 4)
        plt.title('OOD test data (%)', color='RoyalBlue')
        disp = skm.ConfusionMatrixDisplay(confusion_matrix=cm_OOD_normalized, display_labels=classes)
        disp.plot(ax=ax, cmap='binary', values_format=".2f")
        plt.imshow(cm_OOD_normalized, interpolation='nearest', cmap='binary', vmin=min_value, vmax=max_value)

        plt.suptitle('Confusion Matrix', fontsize=23, color='RoyalBlue')
        plt.show()



    def plot_method_micro_stats(self,method):
        """
        Function plots the stats for a given OOD detection method.

        Parameters
        -----------
        method : str
            Name of the OOD detection method
        """
        self.set_style(fontsize=11)
        plt.figure(figsize=(12,12))

        df_ID_method = self.df_ID[self.df_ID['OOD detection method']==method]
        df_OOD_method = self.df_OOD[self.df_OOD['OOD detection method']==method]

        df_ID_method_correct = df_ID_method[df_ID_method['Correct']==True]['Metric']
        df_ID_method_incorrect = df_ID_method[df_ID_method['Correct']==False]['Metric']
        df_OOD_method_correct = df_OOD_method[df_OOD_method['Correct']==True]['Metric']
        df_OOD_method_incorrect = df_OOD_method[df_OOD_method['Correct']==False]['Metric']

        df_ID_method_pred = df_ID_method['Predicted']
        df_OOD_method_pred = df_OOD_method['Predicted']
        df_ID_method_target = df_ID_method['Target']
        df_OOD_method_target = df_OOD_method['Target']

        ID_correct = np.ones(len(df_ID_method))
        OOD_correct = np.zeros(len(df_OOD_method))

        auroc_out_of_dist,aucpr_out_of_dist = self.get_auroc_aucpr(list(ID_correct)+list(OOD_correct), list(df_ID_method['Metric'])+list(df_OOD_method['Metric']))
        auroc_accuracy,aucpr_accuracy = self.get_auroc_aucpr(list(df_ID_method['Correct'])+list(df_OOD_method['Correct']), list(df_ID_method['Metric'])+list(df_OOD_method['Metric']))

        ID_metric = list(df_ID_method['Metric'])
        OOD_metric = list(df_OOD_method['Metric'])

        max_val = np.max([np.max(ID_metric),np.max(OOD_metric)])
        min_val = np.min([np.min(ID_metric),np.min(OOD_metric)])

        ID_metric = (ID_metric - min_val)/(max_val - min_val)
        OOD_metric = (OOD_metric - min_val)/(max_val - min_val)

        df_ID_method_correct = (df_ID_method_correct - min_val)/(max_val-min_val)
        df_ID_method_incorrect = (df_ID_method_incorrect - min_val)/(max_val-min_val)
        df_OOD_method_correct = (df_OOD_method_correct - min_val)/(max_val-min_val)
        df_OOD_method_incorrect = (df_OOD_method_incorrect - min_val)/(max_val-min_val)
        df_ID_method_metric = (np.array(df_ID_method['Metric']) - min_val)/(max_val - min_val)
        df_OOD_method_metric = (np.array(df_OOD_method['Metric']) - min_val)/(max_val - min_val)

        ID_accuracy = []
        OOD_accuracy = []
        ID_bal_accuracy = []
        OOD_bal_accuracy = []
        ID_OOD_accuracy = []
        ID_OOD_bal_accuracy = []

        num_correct = []
        num_total = []

        for threshold in np.arange(0,1+1e-3,1e-3):
            id_mask_correct = sum(df_ID_method_correct >= threshold)
            id_mask_incorrect = sum(df_ID_method_incorrect >= threshold)
            ood_mask_correct = sum(df_OOD_method_correct >= threshold)
            ood_mask_incorrect = sum(df_OOD_method_incorrect >= threshold)

            id_pred_above_thresh = df_ID_method_pred[df_ID_method_metric >= float(threshold)]
            ood_pred_above_thresh =df_OOD_method_pred[df_OOD_method_metric >= float(threshold)]
            id_target_above_thresh = df_ID_method_target[df_ID_method_metric >= float(threshold)]
            ood_target_above_thresh =df_OOD_method_target[df_OOD_method_metric >= float(threshold)]

            ood_sample_weight = [1/len(df_OOD_method) for _ in ood_target_above_thresh]
            id_sample_weight = [1/len(df_ID_method) for _ in id_target_above_thresh]

            ID_bal_accuracy.append(skm.balanced_accuracy_score(id_target_above_thresh,id_pred_above_thresh))
            OOD_bal_accuracy.append(skm.balanced_accuracy_score(ood_target_above_thresh,ood_pred_above_thresh))
            ID_OOD_bal_accuracy.append(skm.balanced_accuracy_score(list(ood_target_above_thresh)+list(id_target_above_thresh),list(ood_pred_above_thresh)+list(id_pred_above_thresh),sample_weight=ood_sample_weight+id_sample_weight))
            ID_accuracy.append(id_mask_correct/(id_mask_correct+id_mask_incorrect) if (id_mask_correct+id_mask_incorrect)!= 0 else np.nan)
            OOD_accuracy.append(ood_mask_correct/(ood_mask_correct+ood_mask_incorrect) if (ood_mask_correct+ood_mask_incorrect)!= 0 else np.nan)

            if np.isnan(ID_accuracy[-1]) == True:
                if np.isnan(OOD_accuracy[-1]) == True:
                    bal_acc = np.nan
                else:
                    bal_acc = OOD_accuracy[-1]
            elif np.isnan(OOD_accuracy[-1]) == True:
                bal_acc = ID_accuracy[-1]
            else:
                bal_acc = (ID_accuracy[-1]+OOD_accuracy[-1])/2
            ID_OOD_accuracy.append(bal_acc)

            num_correct.append(id_mask_correct+ood_mask_correct)
            num_total.append(id_mask_correct+ood_mask_correct+id_mask_incorrect+ood_mask_incorrect)

        plt.subplot(2,2,1)
        self.plot_hist_dist(ID_metric,range=(0,1.0),bins=101,color='RoyalBlue',normalised=True,label='ID')
        self.plot_hist_dist(OOD_metric,range=(0,1.0),bins=101,color='Orange',normalised=True,label='OOD')
        plt.title('OOD AUROC {:.3f}, AUCPR {:.3f}'.format(auroc_out_of_dist,aucpr_out_of_dist),color='grey')
        plt.legend()
        plt.xlabel('Scoring function')
        plt.ylabel('Frequency')
        plt.xlim(0,1)

        plt.subplot(2,2,2)
        self.plot_hist_dist(list(df_ID_method_correct)+list(df_OOD_method_correct),range=(0,1.0),bins=101,color='Green',normalised=True,label='Correct')
        self.plot_hist_dist(list(df_ID_method_incorrect)+list(df_OOD_method_incorrect),range=(0,1.0),bins=101,color='Purple',normalised=True,label='Incorrect')
        plt.title('Accuracy AUROC {:.3f}, AUCPR {:.3f}'.format(auroc_accuracy,aucpr_accuracy),color='grey')
        plt.legend()
        plt.xlabel('Scoring function')
        plt.ylabel('Frequency')
        plt.xlim(0,1)

        plt.subplot(2,2,3)
        plt.plot(np.arange(0,1+1e-3,1e-3),ID_accuracy,color='RoyalBlue',label='ID accuracy')
        plt.plot(np.arange(0,1+1e-3,1e-3),OOD_accuracy,color='Orange',label='OOD accuracy')
        plt.plot(np.arange(0,1+1e-3,1e-3),ID_bal_accuracy,'--',color='RoyalBlue',alpha=0.75,label='ID bal. accuracy')
        plt.plot(np.arange(0,1+1e-3,1e-3),OOD_bal_accuracy,'--',color='Orange',alpha=0.75,label='OOD bal. accuracy')
        plt.plot(np.arange(0,1+1e-3,1e-3),ID_OOD_accuracy,color='grey',label='ID OOD accuracy')
        plt.plot(np.arange(0,1+1e-3,1e-3),ID_OOD_bal_accuracy,'--',color='grey',label='ID OOD bal. accuracy')
        plt.ylabel('Accuracy above threshold')
        plt.legend()
        plt.xlabel('Threshold on scoring function')
        plt.grid(alpha=0.3)
        plt.ylim(0,1)
        plt.xlim(0,1)

        plt.subplot(2,2,4)
        plt.plot(np.arange(0,1+1e-3,1e-3),num_correct,linewidth=2,color='green')
        plt.plot(np.arange(0,1+1e-3,1e-3),num_total,linewidth=2,color='Purple')
        plt.fill_between(np.arange(0,1+1e-3,1e-3),y1=num_correct,y2=np.full(len(num_correct),0),color='green',edgecolor='green',alpha=0.5,label='Correct')
        plt.fill_between(np.arange(0,1+1e-3,1e-3),y1=num_total,y2=num_correct,color='Purple',edgecolor='Purple',alpha=0.5,label='Incorrect')
        plt.legend()
        plt.xlabel('Threshold on scoring function')
        plt.ylabel('# images above threshold')
        plt.grid(alpha=0.3)
        plt.xlim(0,1)
        plt.ylim(0)

        plt.suptitle('OOD detection method: '+str(method),fontsize=23,color='RoyalBlue')
        plt.tight_layout()


    def set_style(self,fontsize=12):
        """
        Sets the style of the plots.

        Parameters
        ----------
        fontsize : int, optional
            The fontsize of the plots. The default is 20.
        """
        plt.rcParams.update({'font.size': fontsize})


PLOT_CHOICES = [
    'accuracy_auc',
    'ood_auc',
    'ood_accuracy_auc',
    'accuracy_aurc',
    'confusion_matrix',
    'method_micro_stats',
]


def create_parser():
    default_output_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description='Analyse and plot micro metric outputs from OOD detection evaluation.'
    )
    parser.add_argument(
        '--output_dir',
        default=default_output_dir,
        type=str,
        help='Directory containing Micro_metrics_ID/OOD*.txt files (default: this script directory).',
    )
    parser.add_argument(
        '--filename',
        default='',
        type=str,
        help='Filename suffix matching evaluate_OOD_detection_method.py --filename, e.g. _my_run (default: empty).',
    )
    parser.add_argument(
        '--method',
        default='all',
        type=str,
        help='OOD detection method name as stored in the metrics file, or "all" (default: all).',
    )
    parser.add_argument(
        '--plot',
        nargs='+',
        choices=PLOT_CHOICES,
        default=['method_micro_stats'],
        help='Plot type(s) to generate. Can pass multiple, e.g. --plot accuracy_auc ood_auc.',
    )
    parser.add_argument(
        '--list_methods',
        action='store_true',
        help='List available OOD detection methods in the metrics files and exit.',
    )
    parser.add_argument(
        '--save_dir',
        default=None,
        type=str,
        help='Optional directory to save figures as PNG files.',
    )
    parser.add_argument(
        '--no_show',
        action='store_true',
        help='Do not display plots interactively (useful with --save_dir).',
    )
    return parser


def _methods_to_plot(visualiser, method):
    if method == 'all':
        return list(visualiser.OOD_detection_methods)
    if method not in visualiser.OOD_detection_methods:
        available = ', '.join(map(str, visualiser.OOD_detection_methods))
        raise ValueError(f'Unknown method "{method}". Available methods: {available}')
    return [method]


def run_plots(visualiser, plots, method):
    methods = _methods_to_plot(visualiser, method)

    for plot_name in plots:
        if plot_name == 'accuracy_auc':
            visualiser.plot_accuracy_AUC(method)
        elif plot_name == 'ood_auc':
            visualiser.plot_OOD_AUC(method)
        elif plot_name == 'ood_accuracy_auc':
            visualiser.plot_OOD_accuracy_AUC(method)
        elif plot_name == 'accuracy_aurc':
            visualiser.plot_accuracy_AURC(method)
        elif plot_name == 'confusion_matrix':
            if method == 'all':
                for plot_method in methods:
                    visualiser.confusion_matrix(method_name=plot_method)
            else:
                visualiser.confusion_matrix(method_name=methods[0])
        elif plot_name == 'method_micro_stats':
            for plot_method in methods:
                visualiser.plot_method_micro_stats(plot_method)
                plt.show()


def main():
    parser = create_parser()
    args = parser.parse_args()

    visualiser = Visualise_outputs(filename=args.filename, output_dir=args.output_dir)

    if args.list_methods:
        print('Available OOD detection methods:')
        for name in visualiser.OOD_detection_methods:
            print(f'  - {name}')
        return

    figure_counter = {'i': 0}
    original_show = plt.show

    def patched_show(*show_args, **show_kwargs):
        if args.save_dir:
            figure_counter['i'] += 1
            os.makedirs(args.save_dir, exist_ok=True)
            path = os.path.join(args.save_dir, f'figure_{figure_counter["i"]:03d}.png')
            plt.savefig(path, bbox_inches='tight', dpi=150)
            print(f'Saved {path}')
            plt.close()
        elif not args.no_show:
            original_show(*show_args, **show_kwargs)

    if args.save_dir or args.no_show:
        plt.show = patched_show

    try:
        run_plots(visualiser, plots=args.plot, method=args.method)
    finally:
        plt.show = original_show


if __name__ == '__main__':
    main()
