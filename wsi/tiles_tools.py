'''
@author: Yang Hu
'''

import math
import os
import sys

from openslide import open_slide

import numpy as np
from support.env import ENV
from support.files import parse_slide_caseid_from_filepath, \
    parse_slideid_from_filepath
from wsi import image_tools, slide_tools, filter_tools
from wsi.filter_tools import tissue_percent


sys.path.append("..")




def tile_to_pil_tile(tile, preload_slide=None):
    """
    Convert tile information into the corresponding tile as a PIL image read from the whole-slide image file.
    
    Args:
      tile: Tile object.
      preload_slide: 
      
      (x, y) -> (w, h) -> (c, r)
    
    Return:
      Tile as a PIL image.
    """
    t = tile
    if preload_slide == None:
        slide_filepath = t.original_slide_filepath
        s = open_slide(slide_filepath)
    else:
        s = preload_slide
    
    x, y = t.large_w_s, t.large_h_s
    w, h = t.large_w_e - t.large_w_s, t.large_h_e - t.large_h_s
    
    tile_region = s.read_region((x, y), 0, (w, h))
    # RGBA to RGB
    pil_img = tile_region.convert("RGB")
    return pil_img


def tile_to_np_tile(tile, preload_slide=None):
    """
    Convert tile information into the corresponding tile as a NumPy image read from the whole-slide image file.
    
    Args:
      tile: Tile object.
    
    Return:
      Tile as a NumPy image.
    """
    pil_img = tile_to_pil_tile(tile, preload_slide)
    np_img = image_tools.pil_to_np_rgb(pil_img)
    return np_img


def generate_tile_image_path(tile, label_type,
                             _env_base_train_dir):
    """
    Obtain tile image path based on tile information such as row, column, row pixel position, column pixel position,
    pixel width, and pixel height.
    
    Args:
      tile: Tile object.
    
    Returns:
      Path to image tile.
    """
    t = tile
#     padded_sl_num = str(t.slide_num).zfill(3)
#     tile_path = os.path.join(TILE_DIR, padded_sl_num,
#                              TRAIN_PREFIX + padded_sl_num + "-" + TILE_SUFFIX + "-r%d-c%d-x%d-y%d-w%d-h%d" % (
#                                t.r, t.c, t.o_c_s, t.o_r_s, t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s) + "." + DEST_TRAIN_EXT)
    
    # get the original slide's filename
    slide_path = t.original_slide_filepath
    slide_case_id = parse_slide_caseid_from_filepath(slide_path)
    slide_type_id = '_' + slide_path[slide_path.find('.') - 3: slide_path.find('.')]
    tile_pos_id = '-(hid' + str(t.h_id) + 'wid' + str(t.w_id) + 'shs' + str(t.small_h_s) + 'she' + str(t.small_h_e) + 'sws' + str(t.small_w_s) + 'swe' + str(t.small_w_e) + ')'
    
    tile_filename = slide_case_id + slide_type_id + tile_pos_id + ENV.PIL_IMAGE_FILE_FORMAT
    
    if label_type == 'emt':
        tile_path = os.path.join(_env_base_train_dir, tile_filename)
    
    return tile_path


def save_display_tile(tile, label_type, _env_base_train_dir,
                      preload_slide=None, save=True, display=False):
    """
    Save and/or display a tile image.
    
    Args:
      tile: Tile object.
      save: If True, save tile image.
      display: If True, dispaly tile image.
    """
    tile_pil_img = tile_to_pil_tile(tile, preload_slide)
    
    if save:
        img_path = generate_tile_image_path(tile, label_type,
                                            _env_base_train_dir)
        tile_dir = os.path.dirname(img_path)
        if not os.path.exists(tile_dir):
            os.makedirs(tile_dir)
        tile_pil_img.save(img_path)
#         print("%-20s | File: %s" % ("Save Tile", img_path))
    
    if display:
        tile_pil_img.show()


class Tile:
    """
    Class for information about a tile.
    
    (h, w) -> (r, c)
    
    
    Components:
        original_slide_filepath: file path original slide for the tile
        np_scaled_tile: np array of the tile in the scaled PIL image
        tile_id: the tile id of this slide
        
        h_id, w_id: the id of tile on height and width
        
        _s: start position
        _e: end position
        o_: of original slide
        
        t_p: percentage of tissue area in tile
        
        _factor: some factor in score counting
        score: quality score of the tile
        
    Functions:
        mask_percentage: the percentage of masked area in this tile
        get_pil_tile: get the PIL image of the tile in original slide
        get_np_tile: get the np array of the tile in original slide
        save_tile: as name
        
        get_pil_scaled_tile: get the PIL image of the tile in scaled slide image
        get_np_scaled_tile: return the 
    
    """
    
    def __init__(self, original_slide_filepath, np_scaled_tile, h_id, w_id, small_h_s, small_h_e, small_w_s, small_w_e, large_h_s, large_h_e, large_w_s,
                 large_w_e, t_p, color_factor, s_and_v_factor, quantity_factor, score):
        self.original_slide_filepath = original_slide_filepath
        self.np_scaled_tile = np_scaled_tile
        self.h_id = h_id
        self.w_id = w_id
        self.small_h_s = small_h_s
        self.small_h_e = small_h_e
        self.small_w_s = small_w_s
        self.small_w_e = small_w_e
        self.large_h_s = large_h_s
        self.large_h_e = large_h_e
        self.large_w_s = large_w_s
        self.large_w_e = large_w_e
        self.tissue_percentage = t_p
        self.color_factor = color_factor
        self.s_and_v_factor = s_and_v_factor
        self.quantity_factor = quantity_factor
        self.score = score
    
    def __str__(self):
        return "[Tile #%s, Row #%d, Column #%d, Small shape: (%d->%d, %d->%d), Large shape: (%d->%d, %d->%d), Tissue %4.2f%%]" % (
          self.original_slide_filepath, self.h_id, self.w_id, self.small_h_s, self.small_h_e,
          self.small_w_s, self.small_w_e, self.large_h_s, self.large_h_e, self.large_w_s, self.large_w_e,
          self.tissue_percentage)
    
    def __repr__(self):
        return "\n" + self.__str__()
    
    def mask_percentage(self):
        return 100 - self.tissue_percentage
        
    def query_caseid(self):
        return parse_slide_caseid_from_filepath(self.original_slide_filepath)
    
    def query_slideid(self):
        """
        slideid = caseid + typeid
        """
        return parse_slideid_from_filepath(self.original_slide_filepath)
         
    def get_pil_tile(self, preload_slide=None):
        return tile_to_pil_tile(self, preload_slide)
    
    def get_np_tile(self, preload_slide=None):
        return tile_to_np_tile(self, preload_slide)
    
    def save_tile(self, label_type, _env_base_train_dir, preload_slide=None):
        save_display_tile(self, label_type, _env_base_train_dir, preload_slide,
                          save=True, display=False)
    
    def get_pil_scaled_tile(self):
        return image_tools.np_to_pil(self.np_scaled_tile)
    
    def get_np_scaled_tile(self):
        return self.np_scaled_tile
    
    def get_pil_scaled_slide(self, scale_factor=ENV.SCALE_FACTOR):
        img, slide = slide_tools.original_slide_and_scaled_pil_image(self.original_slide_filepath,
                                                                     scale_factor,
                                                                     print_opening=False)
        np_img = image_tools.pil_to_np_rgb(img)
        np_img = filter_tools.apply_image_filters(np_img, print_info=False)
        img = image_tools.np_to_pil(np_img)
        return img, slide
    
    def get_np_scaled_slide(self, scale_factor=ENV.SCALE_FACTOR):
        img, slide = slide_tools.original_slide_and_scaled_pil_image(self.original_slide_filepath,
                                                                             scale_factor,
                                                                             print_opening=False)
        np_img = image_tools.pil_to_np_rgb(img)
        np_img = filter_tools.apply_image_filters(np_img, print_info=False)
        return np_img, slide
    
    
def get_num_in_tiles(small_height, small_width, tile_h_size, tile_w_size):
    """
    Obtain the number of vertical and horizontal tiles that an image can be divided into given a row tile size and
    a column tile size.
    
    (h, w) -> (r, c)
    
    Args:
      small_height: height of the scaled slide image.
      small_width: width of the scaled slide image.
      tile_h_size: Number of pixels in a tile row.
      tile_w_size: Number of pixels in a tile column.
      
    Vars:
        num_row_tiles, num_col_tiles: number of tiles on height, number of tiles on width
    
    Returns:
      Tuple consisting of the number of vertical tiles and the number of horizontal tiles that the image can be divided
      into given the row tile size and the column tile size.
    """
    num_row_tiles = math.ceil(small_height / tile_h_size)
    num_col_tiles = math.ceil(small_width / tile_w_size)
    
    return num_row_tiles, num_col_tiles


def get_tile_idxs(small_height, small_width, tile_h_size, tile_w_size):
    """
    Obtain a list of tile coordinates (starting row, ending row, starting column, ending column, row number, column number).
    
    (h, w) -> (r, c)
    
    Args:
      small_height: Number of small_height.
      small_width: Number of columns.
      tile_h_size: tile size of scaled slide image on height.
      tile_w_size: tile size of scaled slide image on width.
      
    Vars:
        num_row_tiles, num_col_tiles: number of tiles on height, number of tiles on width
    
    Returns:
      List of tuples representing tile coordinates consisting of starting row, ending row,
      starting column, ending column, row number, column number.
    """
    idxs = []
    num_row_tiles, num_col_tiles = get_num_in_tiles(small_height, small_width, tile_h_size, tile_w_size)
    
    for h in range(0, num_row_tiles):
        # h is from 0, that's it!
        start_h = h * tile_h_size
        end_h = ((h + 1) * tile_h_size) if (h < num_row_tiles - 1) else small_height
        for w in range(0, num_col_tiles):
            start_w = w * tile_w_size
            end_w = ((w + 1) * tile_w_size) if (w < num_col_tiles - 1) else small_width
            idxs.append((start_h, end_h, start_w, end_w, h + 1, w + 1))
            
    return idxs

def get_overlap_tile_idx():
    """
    Obtain a list of tile coordinates (starting row, ending row, starting column, ending column, row number, column number).
    with overlap fragmenting
    
    (h, w) -> (r, c)
    
    Args:
      small_height: Number of small_height.
      small_width: Number of columns.
      tile_h_size: tile size of scaled slide image on height.
      tile_w_size: tile size of scaled slide image on width.
      overlap_rate: the overlap percentage between two adjacent tiles
      
    Vars:
        num_row_tiles, num_col_tiles: number of tiles on height, number of tiles on width, without overlap
        overlap h, w will cover (1 - overlap_rate) row and col as start for each tile
    
    Returns:
      List of tuples representing overlap tile coordinates consisting of starting row, ending row,
      starting column, ending column, row number, column number.
    """
    pass


def sort_tiles_by_tissue_percentage(tiles):
    """
    """
    sorted_tiles_list = sorted(tiles, key=lambda t: t.tissue_percentage, reverse=True)
    return sorted_tiles_list
    

def get_slide_tiles(np_scaled_img, shape_set_img, original_slide_filepath,
                    _env_tile_w_size, _env_tile_h_size,
                    t_p_threshold=75, load_small_tile=False):
    """
    get all tiles object for a slide
    
    (h, w) -> (r, c)
    
    Args:
        np_scaled_img: np array of the scaled image
        shape_set_img: (large width <of original slide>, large height <of original slide>, 
            small width <of scaled slide image>, small height <of scaled slide image>)
        
        original_slide_filepath: as name
        t_p_threshold: set the threshold of tissue percentage to store
        load_small_tile: if need to load small tile np array in tile object
    
    Returns:
        A list of tiles object for this slide
    """
    
    # (large slide width, large slide height, small slide width, small slide height)
    l_slide_w, l_slide_h, s_slide_w, s_slide_h = shape_set_img
    
    # tile small width size, tile small height size
    t_small_w_size = round(_env_tile_w_size / ENV.SCALE_FACTOR)
    t_small_h_size = round(_env_tile_h_size / ENV.SCALE_FACTOR)
    
    # get the position coordinates (start & end on axis)
    tile_idxs = get_tile_idxs(s_slide_h, s_slide_w, t_small_h_size, t_small_w_size)
    # generate tiles list
    tiles_list = []
    for t_idx in tile_idxs:
        
        small_h_s, small_h_e, small_w_s, small_w_e, h_id, w_id = t_idx
        np_tile = np_scaled_img[small_h_s:small_h_e, small_w_s:small_w_e]
        t_p = filter_tools.tissue_percent(np_tile)
        
        large_w_s, large_h_s = slide_tools.small_to_large_mapping((small_w_s, small_h_s), (l_slide_w, l_slide_h))
        large_w_e, large_h_e = slide_tools.small_to_large_mapping((small_w_e, small_h_e), (l_slide_w, l_slide_h))
        
        # in case the shape out of range
        if (large_w_e - large_w_s) > _env_tile_w_size:
            large_w_e -= 1
        if (large_h_e - large_h_s) > _env_tile_h_size:
            large_h_e -= 1
            
        ''' @TODO'''
        score, color_factor, s_and_v_factor, quantity_factor = None, None, None, None
        
        np_scaled_tile = np_tile if load_small_tile else None
        # produce the tile object
        tile = Tile(original_slide_filepath, np_scaled_tile, h_id, w_id, small_h_s, small_h_e, small_w_s, small_w_e, large_h_s, large_h_e, large_w_s,
                 large_w_e, t_p, color_factor, s_and_v_factor, quantity_factor, score)
        
        if tile.tissue_percentage >= t_p_threshold:
            tiles_list.append(tile)
#     sorted_tiles_list = sort_tiles_by_tissue_percentage(tiles_list)
        
    return tiles_list


if __name__ == '__main__':
    # some test about tile filename generation
    
    '''
    slide_dir = 'D:/TCGA_OV_dataset/example/train_slide'
    slide_filename = 'TCGA-23-1023_DX1.svs'
    slide_filepath = os.path.join(slide_dir, slide_filename)
    
    np_scaled_tile = None
    h_id, w_id, small_h_s, small_h_e, small_w_s, small_w_e, large_h_s, large_h_e, large_w_s, large_w_e = 5, 5, 1, 6, 2, 6, 10, 60, 20, 60
    t_p, color_factor, s_and_v_factor, quantity_factor, score = 0.0, 0.6, 0.6, 0.6, 0.96
    
    tile = Tile(slide_filepath, np_scaled_tile, 
                h_id, w_id, small_h_s, small_h_e, small_w_s, small_w_e, large_h_s, large_h_e, large_w_s, large_w_e,
                t_p, color_factor, s_and_v_factor, quantity_factor, score)
    
    print(generate_tile_image_path(tile))
    '''
    
