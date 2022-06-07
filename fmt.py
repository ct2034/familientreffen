#!/usr/bin/env python3
import argparse
import json
from math import pi
from typing import Dict, Tuple

import numpy as np
import requests
import torch
from matplotlib import pyplot as plt


def address_to_geocoords(address: str) -> Tuple[float, float]:
    url_geocode = f"https://geocode.maps.co/search?q={address}"
    response = requests.get(url_geocode)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to get coordinates for {address},\nNo server response.")
    response_json = response.content
    response_parsed = json.loads(response_json)
    if len(response_parsed) == 0:
        raise RuntimeError(
            f"Failed to get coordinates for {address},\nEmpty response.")
    return (
        float(response_parsed[0]['lat']),
        float(response_parsed[0]['lon'])
    )


def read_input(fname: str) -> Dict[Tuple[float, float], int]:
    out = {}
    data = None
    try:
        with open(fname, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"File {fname} not found.")
    if data is None:
        raise RuntimeError(f"Failed to read {fname}")

    print("Input:")
    for address, n in data.items():
        print(f"  {n} ppl @ {address}")
        coords = address_to_geocoords(address)
        print(f"   {coords}")
        out[coords] = n
    return out


def haversine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == (2,)
    assert b.shape == (2,)

    (lat1, lon1) = a
    (lat2, lon2) = b

    # distance between latitudes
    # and longitudes
    dLat = (b[0] - a[0]) * pi / 180.0
    dLon = (b[1] - a[1]) * pi / 180.0

    # convert to radians
    lat1_rad = a[0] * pi / 180.0
    lat2_rad = b[0] * pi / 180.0

    # apply formulae
    d = (pow(torch.sin(dLat / 2), 2) +
         pow(torch.sin(dLon / 2), 2) *
         torch.cos(lat1_rad) * torch.cos(lat2_rad))
    EARTH_RADIUS_KM = 6371.
    c = 2 * torch.asin(torch.sqrt(d))
    return c * EARTH_RADIUS_KM


def optimize(coords, centre):
    learning_rate = 1E-6
    n_optim = 100
    n_coords = coords.shape[0]
    dists = torch.zeros(n_coords)

    poss = []
    loss_s = []

    for _ in range(n_optim):
        for i in range(n_coords):
            dists[i] = haversine(coords[i], centre)
        loss = dists.pow(2).sum()
        loss_s.append(loss.item())

        # Backprop to compute gradients of a, b, c, d with respect to loss
        _ = loss.backward(retain_graph=True)
        # print(x)
        # print(x.grad)
        centre = (centre - (centre.grad * learning_rate)
                  ).clone().detach().requires_grad_(True)
        poss.append(centre.tolist())

    # plt.plot(np.array(poss)[:, 0], np.array(poss)[:, 1], '-o')
    # plt.plot(np.array(poss)[-1, 0], np.array(poss)[-1, 1], 'xr')
    # plt.figure()
    # plt.plot(np.array(loss_s))
    # plt.show()

    return centre


def adress_from_lat_lon(lat: float, lon: float) -> str:
    url_geocode = f"https://geocode.maps.co/reverse?lat={lat}&lon={lon}"
    response = requests.get(url_geocode)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to get coordinates for {lat}, {lon},\nNo server response.")
    response_json = response.content
    response_parsed = json.loads(response_json)
    if len(response_parsed) == 0:
        raise RuntimeError(
            f"Failed to get coordinates for {lat}, {lon},\nEmpty response.")
    return response_parsed['display_name']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute centre of locations')
    parser.add_argument('fname', type=str, help='json file with addresses')
    args = parser.parse_args()

    # Read Inputs
    coords = read_input(args.fname)
    print(coords)

    # Prepare Data
    coords_cumm = []
    for coord, n in coords.items():
        assert n > 0
        for _ in range(n):
            coords_cumm.append(coord)
    assert len(coords_cumm) > 0
    coords_torch = torch.tensor(coords_cumm, dtype=float)

    # Outputs
    print("\nOutputs:")

    # mean
    centre_mean = torch.mean(coords_torch, axis=0)
    print("  mean:")
    print(f"   {centre_mean.tolist()}")
    print(f"    {adress_from_lat_lon(centre_mean[0], centre_mean[1])}")

    # geometric middle
    centre_geo = centre_mean.clone().detach().requires_grad_(True)
    centre_geo = optimize(coords_torch, centre_geo)
    print("  geometric centre:")
    print(f"   {centre_geo.tolist()}")
    print(f"    {adress_from_lat_lon(centre_geo[0], centre_geo[1])}")

    plt.figure()
    plt.plot(coords_torch.detach().numpy()[:, 0],
             coords_torch.detach().numpy()[:, 1],
             'og')
    plt.plot(centre_mean.detach().numpy()[0],
             centre_mean.detach().numpy()[1],
             'xr')
    plt.plot(centre_geo.detach().numpy()[0],
             centre_geo.detach().numpy()[1],
             'xr')
    plt.axis('equal')
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    plt.show()
