#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <string.h>
#include <float.h>
#include <wchar.h>

#define FEATURE_NUMBER 128
#define NAME_LENGTH 64

typedef struct PersonFeature{
    wchar_t name[NAME_LENGTH];
    double right[FEATURE_NUMBER];
    double left[FEATURE_NUMBER];
    double front[FEATURE_NUMBER];
} PersonFeature;

typedef struct PersonFeatureList{
    PersonFeature feature;
    struct PersonFeatureList *next;
} PersonFeatureList;

PersonFeatureList *g_feature_list;

static int save_list() {
    FILE *file;
    PersonFeatureList *tmp = g_feature_list;

    file = fopen("feature.data", "wb");

    while (tmp != NULL) {
        fwrite(&(tmp->feature), sizeof(PersonFeature), 1, file);
        tmp = tmp->next;
    }
    fclose(file);
}

int delete_name(wchar_t name[]) {
    int found = 0;
    PersonFeatureList *tmp = g_feature_list, *before;

    while(tmp != NULL) {
       if (wcsncmp(name, tmp->feature.name, wcslen(name)) == 0) {
           found = 1;
           break;
       }
       before = tmp;
       tmp = tmp->next;
    }

    if (found) {
        if(tmp == g_feature_list)
            g_feature_list = g_feature_list->next;
        else
            before->next = tmp->next;
        free(tmp);
        save_list();
    }
    return found;
}

int save_feature(wchar_t name[], double right[], double left[], double front[]){
    PersonFeatureList *tmp;
    FILE *file;

    if (g_feature_list == NULL)
        g_feature_list = tmp = malloc(sizeof(PersonFeature));
    else{
        tmp = g_feature_list;
        while(tmp->next != NULL)
            tmp = tmp->next;
        tmp->next = malloc(sizeof(PersonFeature));
        tmp = tmp->next;
    }
        
    tmp->next = NULL;
    memcpy(tmp->feature.name, name, sizeof(wchar_t) * (wcslen(name) + 1));
    memcpy(tmp->feature.right, right, sizeof(double) * FEATURE_NUMBER);
    memcpy(tmp->feature.left, left, sizeof(double) * FEATURE_NUMBER);
    memcpy(tmp->feature.front, front, sizeof(double) * FEATURE_NUMBER);

    file = fopen("feature.data", "ab");
    fwrite(&(tmp->feature), sizeof(PersonFeature), 1, file);
    fclose(file);

    return 0;
}

int load_feature()
{
    PersonFeatureList *tmp, *before;
    FILE *file;

    if (g_feature_list != NULL) {
        printf("Feature have been loaded!\n");
        goto out;
    }
    file = fopen("feature.data", "rb");
    if (file == NULL) {
        printf("Open feature data file error!\n");
        goto out;
    }

    g_feature_list = tmp = before = malloc(sizeof(PersonFeature));
    while(fread(&(tmp->feature), sizeof(PersonFeature), 1, file) == 1) {
        tmp->next = malloc(sizeof(PersonFeature));
        before = tmp;
        tmp = tmp->next;
    }

    if (g_feature_list == tmp)
        g_feature_list = NULL;

    before->next = NULL;
    free(tmp);
    fclose(file);

    printf("Load feature successfully!\n");

    return 0;
out:
    return -1;
}

#define POS_RIGHT 0
#define POS_LEFT  1
#define POS_FROUT 2

wchar_t* find_people(double feature[], int pos, double thres, int percent_thres)
{
    double *data, smallest = DBL_MAX;
    PersonFeatureList *ret = NULL, *tmp;

    tmp = g_feature_list;
    while(tmp != NULL) {
        switch (pos){
            case POS_RIGHT:
                data = tmp->feature.right;
                break;
            case POS_LEFT:
                data = tmp->feature.left;
                break;
            case POS_FROUT:
                data = tmp->feature.front;
                break;
            default:
                goto out;
        }

        double distance = 0;
        for (int i = 0; i < FEATURE_NUMBER; i++) {
            double tmp;
            tmp = data[i] - feature[i];
            distance += (tmp * tmp);
        }
        distance = sqrt(distance);
        if(distance < smallest) {
            smallest = distance;
            ret = tmp;
        }
        tmp = tmp->next;
    }
    if (percent_thres < 100 * thres / smallest) {
        return ret->feature.name;
    }
out:
    return NULL;
}

void compare_feature(double m[], double n[])
{
   double distance = 0;
   for (int i = 0; i < FEATURE_NUMBER; i++) {
        double tmp;
        tmp = (m[i] - n[i]) * (m[i] - n[i]);
        distance += tmp;
    }
    printf("C distance:%.10f", sqrt(distance));
}
